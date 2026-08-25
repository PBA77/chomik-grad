from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .lazy import (
    Compiler,
    CompiledProgram,
    DeviceAdapter,
    LazyNode,
    Op,
    topological_sort,
)


@dataclass(frozen=True)
class VulkanArray:
    buffer: Any
    shape: Tuple[int, ...]
    dtype: np.dtype
    strides: Tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))


class VulkanDeviceAdapter(DeviceAdapter):
    name = "vulkan"

    def __init__(self, wgpu: Any, device: Any) -> None:
        self._wgpu = wgpu
        self.device = device
        self.queue = device.queue

    @staticmethod
    def _c_strides(shape: Sequence[int]) -> Tuple[int, ...]:
        strides = []
        running = 1
        for size in reversed(shape):
            strides.append(running)
            running *= int(size)
        return tuple(reversed(strides))

    def array(self, value: object) -> VulkanArray:
        array = np.asarray(value)
        if array.dtype == np.int64:
            limits = np.iinfo(np.int32)
            if array.size and (
                int(array.min()) < limits.min or int(array.max()) > limits.max
            ):
                raise OverflowError("Vulkan indices must fit in int32")
            array = array.astype(np.int32)
        if array.dtype not in (np.float32, np.int32):
            raise TypeError(
                "the Vulkan FP32 backend supports float32 tensors and "
                "int32-compatible gather indices"
            )
        array = np.ascontiguousarray(array)
        upload = array if array.nbytes else np.zeros(1, dtype=array.dtype)
        usage = (
            self._wgpu.BufferUsage.STORAGE
            | self._wgpu.BufferUsage.COPY_SRC
            | self._wgpu.BufferUsage.COPY_DST
        )
        buffer = self.device.create_buffer_with_data(data=upload, usage=usage)
        return VulkanArray(
            buffer,
            tuple(array.shape),
            np.dtype(array.dtype),
            self._c_strides(array.shape),
        )

    def evaluate(self, values: Sequence[object]) -> None:
        self.synchronize()

    def synchronize(self) -> None:
        self.queue.on_submitted_work_done_sync()

    def to_numpy(self, value: object) -> np.ndarray:
        native = value
        raw = self.queue.read_buffer(native.buffer)  # type: ignore[attr-defined]
        dtype = np.dtype(native.dtype)  # type: ignore[attr-defined]
        strides = tuple(  # type: ignore[attr-defined]
            int(stride) * dtype.itemsize for stride in native.strides
        )
        view = np.ndarray(
            native.shape,  # type: ignore[attr-defined]
            dtype=dtype,
            buffer=raw,
            strides=strides,
        )
        return np.array(view, copy=True)

    def argmax(self, value: object) -> int:
        return int(self.to_numpy(value).argmax())

    def argmax_last_axis(self, value: object) -> np.ndarray:
        return self.to_numpy(value).argmax(axis=-1)

    def dtype(self, value: object) -> np.dtype:
        return np.dtype(value.dtype)  # type: ignore[attr-defined]


@dataclass
class VulkanProgram(CompiledProgram):
    source: str
    inputs: Tuple[LazyNode, ...]
    _run: Callable[[Sequence[VulkanArray], Any], Tuple[VulkanArray, ...]]
    device: VulkanDeviceAdapter
    _load_input: Callable[[LazyNode], VulkanArray]

    def run(
        self,
        bindings: Optional[Mapping[LazyNode, object]] = None,
        *,
        synchronize: bool = False,
    ) -> Tuple[object, ...]:
        replacements = bindings or {}
        values = [
            replacements[node] if node in replacements else self._load_input(node)
            for node in self.inputs
        ]
        encoder = self.device.device.create_command_encoder()
        outputs = tuple(self._run(values, encoder))  # type: ignore[arg-type]
        self.device.queue.submit([encoder.finish()])
        if synchronize:
            self.device.evaluate(outputs)
        return outputs


class VulkanCompiler(Compiler):
    """Execute the portable tensor IR as WGSL compute on Vulkan."""

    _BINARY_EXPRESSIONS = {
        "add": "left_value + right_value",
        "sub": "left_value - right_value",
        "mul": "left_value * right_value",
        "div": "left_value / right_value",
        "equal": "select(0.0, 1.0, left_value == right_value)",
    }
    _UNARY_EXPRESSIONS = {
        "identity": "input_value",
        "neg": "-input_value",
        "exp": "exp(input_value)",
        "log": "log(input_value)",
        "sqrt": "sqrt(input_value)",
        "relu": "max(input_value, 0.0)",
        "step": "select(0.0, 1.0, input_value > 0.0)",
    }

    def __init__(self) -> None:
        try:
            import wgpu
        except ImportError as error:
            raise ImportError(
                "the Vulkan backend requires `python -m pip install "
                "'chomik-grad[vulkan]'`"
            ) from error

        adapters = [
            adapter
            for adapter in wgpu.gpu.enumerate_adapters_sync()
            if str(adapter.info.get("backend_type", "")).lower() == "vulkan"
        ]
        if not adapters:
            raise RuntimeError("the Vulkan backend requires a Vulkan adapter")
        adapters.sort(
            key=lambda adapter: adapter.info.get("adapter_type")
            != "DiscreteGPU"
        )
        self._wgpu = wgpu
        self.adapter = adapters[0]
        self.adapter_info = dict(self.adapter.info)
        self.device_name = str(self.adapter_info.get("device", "Vulkan device"))
        self._device = self.adapter.request_device_sync()
        self.device = VulkanDeviceAdapter(wgpu, self._device)
        self._program_cache: Dict[object, Tuple[str, Callable[..., Any]]] = {}
        self._pipeline_cache: Dict[object, Tuple[Any, Any]] = {}

    @property
    def cache_size(self) -> int:
        return len(self._program_cache)

    def close(self) -> None:
        self.device.synchronize()
        self._program_cache.clear()
        self._pipeline_cache.clear()

    @staticmethod
    def _c_strides(shape: Sequence[int]) -> Tuple[int, ...]:
        return VulkanDeviceAdapter._c_strides(shape)

    @classmethod
    def _is_c_contiguous(cls, value: VulkanArray) -> bool:
        expected = 1
        for size, stride in zip(reversed(value.shape), reversed(value.strides)):
            if size > 1 and stride != expected:
                return False
            expected *= size
        return True

    def _load_input(self, node: LazyNode) -> VulkanArray:
        if "vulkan" in node.native_values:
            return node.native_values["vulkan"]
        value = node.numpy_value()
        if value.dtype not in (np.float32, np.int32, np.int64):
            raise TypeError(
                "the Vulkan FP32 backend supports float32 tensors and "
                "int32-compatible gather indices"
            )
        native = self.device.array(value)
        if node.cache_native:
            node.native_values["vulkan"] = native
        return native

    @staticmethod
    def _signature(
        nodes: Sequence[LazyNode], outputs: Sequence[LazyNode]
    ) -> object:
        indexes = {node: index for index, node in enumerate(nodes)}
        specifications = []
        for node in nodes:
            if node.op is None:
                specifications.append((None, node.shape, node.dtype.str))
            else:
                specifications.append(
                    (
                        node.op.value,
                        node.arg,
                        node.shape,
                        node.dtype.str,
                        tuple(indexes[parent] for parent in node.inputs),
                    )
                )
        return tuple(specifications), tuple(indexes[node] for node in outputs)

    def compile(
        self,
        outputs: Sequence[LazyNode],
        dynamic_inputs: Sequence[LazyNode] = (),
    ) -> VulkanProgram:
        if not outputs:
            raise ValueError("at least one output is required")
        if any(node.op is not None for node in dynamic_inputs):
            raise ValueError("dynamic inputs must be graph leaves")

        nodes = topological_sort(outputs)
        leaves = tuple(node for node in nodes if node.op is None)
        requested_dynamic = set(dynamic_inputs)
        if not requested_dynamic.issubset(leaves):
            raise ValueError("dynamic input is not a leaf of the compiled graph")
        program_inputs = tuple(node for node in leaves if node in requested_dynamic)
        specialized = bool(dynamic_inputs)
        signature = self._signature(nodes, outputs)
        cached = None if specialized else self._program_cache.get(signature)
        if cached is not None:
            source, run = cached
            return VulkanProgram(source, leaves, run, self.device, self._load_input)

        names = {node: f"v{index}" for index, node in enumerate(nodes)}
        leaf_indexes = {node: index for index, node in enumerate(leaves)}
        last_uses: Dict[LazyNode, int] = {}
        for index, node in enumerate(nodes):
            for parent in node.inputs:
                last_uses[parent] = index
        for output in outputs:
            last_uses[output] = len(nodes)
        releases: Dict[int, list[str]] = {}
        for node, index in last_uses.items():
            if index < len(nodes):
                releases.setdefault(index, []).append(names[node])

        lines = ["def run(inputs, encoder):"]
        for index, node in enumerate(nodes):
            name = names[node]
            if node.op is None:
                lines.append(f"    {name} = inputs[{leaf_indexes[node]}]")
                continue
            args = [names[parent] for parent in node.inputs]
            lines.append(f"    {name} = {self._expression(node, args)}")
            released = releases.get(index)
            if released:
                lines.append(f"    del {', '.join(released)}")

        rendered_outputs = ", ".join(names[node] for node in outputs)
        if len(outputs) == 1:
            rendered_outputs += ","
        lines.append(f"    return ({rendered_outputs})")
        source = "\n".join(lines)

        namespace: Dict[str, object] = {}
        exec(
            compile(source, "<chomikgrad-vulkan>", "exec"),
            {
                "binary": self._binary,
                "gather": self._gather,
                "matmul": self._matmul,
                "permute": self._permute,
                "reduce": self._reduce,
                "reshape": self._reshape,
                "unary": self._unary,
            },
            namespace,
        )
        raw_run = namespace["run"]
        if specialized:
            dynamic_indexes = {
                node: index for index, node in enumerate(program_inputs)
            }
            template = [
                None if node in requested_dynamic else self._load_input(node)
                for node in leaves
            ]

            def run(values: Sequence[Any], encoder: Any) -> Tuple[Any, ...]:
                merged = list(template)
                for leaf_index, node in enumerate(leaves):
                    if node in dynamic_indexes:
                        merged[leaf_index] = values[dynamic_indexes[node]]
                return raw_run(merged, encoder)

        else:
            run = raw_run
            self._program_cache[signature] = source, run

        return VulkanProgram(  # type: ignore[arg-type]
            source,
            program_inputs if specialized else leaves,
            run,
            self.device,
            self._load_input,
        )

    def _expression(self, node: LazyNode, args: Sequence[str]) -> str:
        if node.dtype != np.dtype(np.float32) and node.op is not Op.GATHER:
            raise TypeError("the Vulkan backend currently supports FP32 operations")
        if node.op is Op.ELEMENTWISE:
            kind = str(node.arg)
            if kind in self._BINARY_EXPRESSIONS:
                return (
                    f"binary(encoder, {kind!r}, {args[0]}, {args[1]}, "
                    f"{node.shape!r})"
                )
            if kind in self._UNARY_EXPRESSIONS:
                return f"unary(encoder, {kind!r}, {args[0]}, {node.shape!r})"
            raise ValueError(f"unsupported elementwise operation: {kind}")
        if node.op is Op.REDUCE:
            kind, axes, keepdims = node.arg  # type: ignore[misc]
            return (
                f"reduce(encoder, {kind!r}, {args[0]}, {tuple(axes)!r}, "
                f"{bool(keepdims)!r}, {node.shape!r})"
            )
        if node.op is Op.RESHAPE:
            return f"reshape(encoder, {args[0]}, {node.shape!r})"
        if node.op is Op.PERMUTE:
            return f"permute({args[0]}, {tuple(node.arg)!r})"
        if node.op is Op.MATMUL:
            return f"matmul(encoder, {args[0]}, {args[1]}, {node.shape!r})"
        if node.op is Op.GATHER:
            return f"gather(encoder, {args[0]}, {args[1]}, {node.shape!r})"
        raise ValueError(f"unsupported operation: {node.op}")

    def _empty(
        self, shape: Tuple[int, ...], dtype: np.dtype = np.dtype(np.float32)
    ) -> VulkanArray:
        dtype = np.dtype(dtype)
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        usage = (
            self._wgpu.BufferUsage.STORAGE
            | self._wgpu.BufferUsage.COPY_SRC
            | self._wgpu.BufferUsage.COPY_DST
        )
        buffer = self._device.create_buffer(size=max(4, size), usage=usage)
        return VulkanArray(buffer, shape, dtype, self._c_strides(shape))

    def _build_pipeline(
        self,
        key: object,
        source: str,
        access: Tuple[str, ...],
    ) -> Tuple[Any, Any]:
        cached = self._pipeline_cache.get(key)
        if cached is not None:
            return cached
        entries = []
        for binding, mode in enumerate(access):
            binding_type = (
                self._wgpu.BufferBindingType.storage
                if mode == "write"
                else self._wgpu.BufferBindingType.read_only_storage
            )
            entries.append(
                {
                    "binding": binding,
                    "visibility": self._wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": binding_type,
                        "has_dynamic_offset": False,
                    },
                }
            )
        layout = self._device.create_bind_group_layout(entries=entries)
        pipeline_layout = self._device.create_pipeline_layout(
            bind_group_layouts=[layout]
        )
        module = self._device.create_shader_module(code=source)
        pipeline = self._device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": module, "entry_point": "main"},
        )
        self._pipeline_cache[key] = layout, pipeline
        return layout, pipeline

    def _dispatch(
        self,
        encoder: Any,
        key: object,
        source: str,
        values: Sequence[VulkanArray],
        access: Tuple[str, ...],
        workgroups: Tuple[int, int, int],
    ) -> None:
        layout, pipeline = self._build_pipeline(key, source, access)
        entries = [
            {
                "binding": binding,
                "resource": {
                    "buffer": value.buffer,
                    "offset": 0,
                    "size": value.buffer.size,
                },
            }
            for binding, value in enumerate(values)
        ]
        bind_group = self._device.create_bind_group(
            layout=layout, entries=entries
        )
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(*workgroups)
        compute_pass.end()

    @staticmethod
    def _linear_workgroups(total: int, width: int = 256) -> Tuple[int, int, int]:
        groups = (total + width - 1) // width
        groups_x = min(groups, 65535)
        return groups_x, (groups + groups_x - 1) // groups_x, 1

    @classmethod
    def _broadcast_offset_expression(
        cls,
        value: VulkanArray,
        output_shape: Tuple[int, ...],
        index_name: str,
    ) -> str:
        rank = len(output_shape)
        padding = rank - value.ndim
        shapes = (1,) * padding + value.shape
        strides = (0,) * padding + value.strides
        output_strides = cls._c_strides(output_shape)
        terms = ["0u"]
        for axis, (size, stride) in enumerate(zip(shapes, strides)):
            if size != 1 and stride:
                terms.append(
                    f"((({index_name} / {output_strides[axis]}u) % "
                    f"{output_shape[axis]}u) * {stride}u)"
                )
        return " + ".join(terms)

    def _binary(
        self,
        encoder: Any,
        kind: str,
        left: VulkanArray,
        right: VulkanArray,
        output_shape: Tuple[int, ...],
    ) -> VulkanArray:
        if left.dtype != np.float32 or right.dtype != np.float32:
            raise TypeError("Vulkan elementwise operations require float32")
        output = self._empty(output_shape)
        total = output.size
        if total == 0:
            return output
        workgroups = self._linear_workgroups(total)
        row_width = workgroups[0] * 256
        left_offset = self._broadcast_offset_expression(left, output_shape, "gid")
        right_offset = self._broadcast_offset_expression(
            right, output_shape, "gid"
        )
        expression = self._BINARY_EXPRESSIONS[kind]
        source = f"""
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let gid = global_id.x + group_id.y * {row_width}u;
    if (gid >= {total}u) {{ return; }}
    let left_value = left[{left_offset}];
    let right_value = right[{right_offset}];
    output[gid] = {expression};
}}
"""
        key = (
            "binary",
            kind,
            left.shape,
            left.strides,
            right.shape,
            right.strides,
            output_shape,
        )
        self._dispatch(
            encoder,
            key,
            source,
            (left, right, output),
            ("read", "read", "write"),
            workgroups,
        )
        return output

    def _unary(
        self,
        encoder: Any,
        kind: str,
        value: VulkanArray,
        output_shape: Tuple[int, ...],
    ) -> VulkanArray:
        if value.dtype != np.float32:
            raise TypeError("Vulkan elementwise operations require float32")
        output = self._empty(output_shape)
        total = output.size
        if total == 0:
            return output
        workgroups = self._linear_workgroups(total)
        row_width = workgroups[0] * 256
        input_offset = self._broadcast_offset_expression(
            value, output_shape, "gid"
        )
        expression = self._UNARY_EXPRESSIONS[kind]
        source = f"""
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let gid = global_id.x + group_id.y * {row_width}u;
    if (gid >= {total}u) {{ return; }}
    let input_value = input_values[{input_offset}];
    output[gid] = {expression};
}}
"""
        key = ("unary", kind, value.shape, value.strides, output_shape)
        self._dispatch(
            encoder,
            key,
            source,
            (value, output),
            ("read", "write"),
            workgroups,
        )
        return output

    def _reduce(
        self,
        encoder: Any,
        kind: str,
        value: VulkanArray,
        axes: Tuple[int, ...],
        keepdims: bool,
        output_shape: Tuple[int, ...],
    ) -> VulkanArray:
        del keepdims
        if value.dtype != np.float32:
            raise TypeError("Vulkan reductions require float32")
        if kind not in ("sum", "max"):
            raise ValueError(f"unsupported Vulkan reduction: {kind}")
        output = self._empty(output_shape)
        output_count = output.size
        if output_count == 0:
            return output
        retained = tuple(axis for axis in range(value.ndim) if axis not in axes)
        reduced_shapes = tuple(value.shape[axis] for axis in axes)
        retained_shapes = tuple(value.shape[axis] for axis in retained)
        reduced_strides = self._c_strides(reduced_shapes)
        retained_strides = self._c_strides(retained_shapes)
        terms = ["0u"]
        for index, axis in enumerate(retained):
            terms.append(
                f"(((output_index / {retained_strides[index]}u) % "
                f"{retained_shapes[index]}u) * {value.strides[axis]}u)"
            )
        for index, axis in enumerate(axes):
            terms.append(
                f"(((reduction_index / {reduced_strides[index]}u) % "
                f"{reduced_shapes[index]}u) * {value.strides[axis]}u)"
            )
        input_index = " + ".join(terms)
        neutral = "0.0" if kind == "sum" else "-3.402823466e+38"
        combine = (
            "accumulator = accumulator + current;"
            if kind == "sum"
            else "accumulator = max(accumulator, current);"
        )
        merge = (
            "scratch[lane] = scratch[lane] + scratch[lane + stride];"
            if kind == "sum"
            else "scratch[lane] = max(scratch[lane], scratch[lane + stride]);"
        )
        reduction_count = int(np.prod(reduced_shapes, dtype=np.int64))
        workgroups = (min(output_count, 65535), 1, 1)
        if output_count > 65535:
            workgroups = (
                65535,
                (output_count + 65534) // 65535,
                1,
            )
        groups_x = workgroups[0]
        source = f"""
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let output_index = group_id.x + group_id.y * {groups_x}u;
    if (output_index >= {output_count}u) {{ return; }}
    let lane = local_id.x;
    var accumulator = {neutral};
    var reduction_index = lane;
    loop {{
        if (reduction_index >= {reduction_count}u) {{ break; }}
        let current = input_values[{input_index}];
        {combine}
        reduction_index = reduction_index + 256u;
    }}
    scratch[lane] = accumulator;
    workgroupBarrier();
    var stride = 128u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lane < stride) {{ {merge} }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    if (lane == 0u) {{ output[output_index] = scratch[0]; }}
}}
"""
        key = (
            "reduce",
            kind,
            value.shape,
            value.strides,
            axes,
            output_shape,
        )
        self._dispatch(
            encoder,
            key,
            source,
            (value, output),
            ("read", "write"),
            workgroups,
        )
        return output

    def _contiguous(self, encoder: Any, value: VulkanArray) -> VulkanArray:
        if self._is_c_contiguous(value):
            return value
        return self._unary(encoder, "identity", value, value.shape)

    def _reshape(
        self, encoder: Any, value: VulkanArray, shape: Tuple[int, ...]
    ) -> VulkanArray:
        contiguous = self._contiguous(encoder, value)
        return VulkanArray(
            contiguous.buffer,
            shape,
            contiguous.dtype,
            self._c_strides(shape),
        )

    @staticmethod
    def _permute(value: VulkanArray, axes: Tuple[int, ...]) -> VulkanArray:
        return VulkanArray(
            value.buffer,
            tuple(value.shape[axis] for axis in axes),
            value.dtype,
            tuple(value.strides[axis] for axis in axes),
        )

    @classmethod
    def _batch_offset_expression(
        cls,
        value: VulkanArray,
        output_batch_shape: Tuple[int, ...],
        index_name: str,
    ) -> str:
        padding = len(output_batch_shape) - (value.ndim - 2)
        shapes = (1,) * padding + value.shape[:-2]
        strides = (0,) * padding + value.strides[:-2]
        output_strides = cls._c_strides(output_batch_shape)
        terms = ["0u"]
        for axis, (size, stride) in enumerate(zip(shapes, strides)):
            if size != 1:
                terms.append(
                    f"((({index_name} / {output_strides[axis]}u) % "
                    f"{output_batch_shape[axis]}u) * {stride}u)"
                )
        return " + ".join(terms)

    def _matmul(
        self,
        encoder: Any,
        left: VulkanArray,
        right: VulkanArray,
        output_shape: Tuple[int, ...],
    ) -> VulkanArray:
        if left.dtype != np.float32 or right.dtype != np.float32:
            raise TypeError("Vulkan matmul requires float32")
        output = self._empty(output_shape)
        if output.size == 0:
            return output
        batch_shape = output_shape[:-2]
        batch_count = int(np.prod(batch_shape, dtype=np.int64))
        m, n = output_shape[-2:]
        k = left.shape[-1]
        left_batch = self._batch_offset_expression(left, batch_shape, "batch")
        right_batch = self._batch_offset_expression(right, batch_shape, "batch")
        source = f"""
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
var<workgroup> left_tile: array<f32, 256>;
var<workgroup> right_tile: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let column = global_id.x;
    let row = global_id.y;
    let batch = global_id.z;
    let lane = local_id.y * 16u + local_id.x;
    let left_base = {left_batch};
    let right_base = {right_batch};
    var accumulator = 0.0;
    var tile = 0u;
    loop {{
        let left_column = tile * 16u + local_id.x;
        let right_row = tile * 16u + local_id.y;
        if (row < {m}u && left_column < {k}u) {{
            left_tile[lane] = left[
                left_base + row * {left.strides[-2]}u
                + left_column * {left.strides[-1]}u
            ];
        }} else {{
            left_tile[lane] = 0.0;
        }}
        if (right_row < {k}u && column < {n}u) {{
            right_tile[lane] = right[
                right_base + right_row * {right.strides[-2]}u
                + column * {right.strides[-1]}u
            ];
        }} else {{
            right_tile[lane] = 0.0;
        }}
        workgroupBarrier();
        var inner = 0u;
        loop {{
            if (inner >= 16u) {{ break; }}
            accumulator = accumulator
                + left_tile[local_id.y * 16u + inner]
                * right_tile[inner * 16u + local_id.x];
            inner = inner + 1u;
        }}
        workgroupBarrier();
        tile = tile + 1u;
        if (tile * 16u >= {k}u) {{ break; }}
    }}
    if (row < {m}u && column < {n}u) {{
        output[batch * {m * n}u + row * {n}u + column] = accumulator;
    }}
}}
"""
        key = (
            "matmul",
            left.shape,
            left.strides,
            right.shape,
            right.strides,
            output_shape,
        )
        workgroups = ((n + 15) // 16, (m + 15) // 16, batch_count)
        self._dispatch(
            encoder,
            key,
            source,
            (left, right, output),
            ("read", "read", "write"),
            workgroups,
        )
        return output

    def _gather(
        self,
        encoder: Any,
        source: VulkanArray,
        indices: VulkanArray,
        output_shape: Tuple[int, ...],
    ) -> VulkanArray:
        if source.dtype != np.float32 or indices.dtype != np.int32:
            raise TypeError("Vulkan gather requires float32 values and int32 indices")
        source = self._contiguous(encoder, source)
        indices = self._contiguous(encoder, indices)
        output = self._empty(output_shape)
        total = output.size
        if total == 0:
            return output
        width = int(np.prod(source.shape[1:], dtype=np.int64))
        workgroups = self._linear_workgroups(total)
        row_width = workgroups[0] * 256
        shader = f"""
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let gid = global_id.x + group_id.y * {row_width}u;
    if (gid >= {total}u) {{ return; }}
    let item = gid / {width}u;
    let column = gid % {width}u;
    let row = u32(indices[item]);
    output[gid] = source[row * {width}u + column];
}}
"""
        key = ("gather", source.shape, indices.shape)
        self._dispatch(
            encoder,
            key,
            shader,
            (source, indices, output),
            ("read", "read", "write"),
            workgroups,
        )
        return output

    def update_parameters(
        self,
        parameters: Sequence[LazyNode],
        gradients: Sequence[LazyNode],
        learning_rate: float,
        *,
        inplace: bool = False,
    ) -> Tuple[Tuple[LazyNode, ...], Tuple[LazyNode, ...]]:
        gradient_program = self.compile(gradients)
        native_gradients = gradient_program.run(synchronize=False)
        native_parameters = [self._load_input(node) for node in parameters]
        encoder = self._device.create_command_encoder()
        updated = tuple(
            self._update_parameter(
                encoder,
                parameter,  # type: ignore[arg-type]
                gradient,  # type: ignore[arg-type]
                learning_rate,
                inplace=inplace,
            )
            for parameter, gradient in zip(native_parameters, native_gradients)
        )
        self.device.queue.submit([encoder.finish()])
        parameter_nodes = tuple(
            LazyNode.native_leaf("vulkan", value, node.shape, node.dtype)
            for node, value in zip(parameters, updated)
        )
        gradient_nodes = tuple(
            LazyNode.native_leaf("vulkan", value, node.shape, node.dtype)
            for node, value in zip(parameters, native_gradients)
        )
        return parameter_nodes, gradient_nodes

    def _update_parameter(
        self,
        encoder: Any,
        parameter: VulkanArray,
        gradient: VulkanArray,
        learning_rate: float,
        *,
        inplace: bool,
    ) -> VulkanArray:
        parameter = self._contiguous(encoder, parameter)
        gradient = self._contiguous(encoder, gradient)
        if parameter.dtype != np.float32 or gradient.dtype != np.float32:
            raise TypeError("Vulkan SGD requires float32")
        output = parameter if inplace else self._empty(parameter.shape)
        total = parameter.size
        if total == 0:
            return output
        workgroups = self._linear_workgroups(total)
        row_width = workgroups[0] * 256
        if inplace:
            source = f"""
@group(0) @binding(0) var<storage, read_write> parameter: array<f32>;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let gid = global_id.x + group_id.y * {row_width}u;
    if (gid < {total}u) {{
        parameter[gid] = parameter[gid] - {float(learning_rate)!r} * gradient[gid];
    }}
}}
"""
            self._dispatch(
                encoder,
                ("sgd", "inplace", total, float(learning_rate)),
                source,
                (parameter, gradient),
                ("write", "read"),
                workgroups,
            )
        else:
            source = f"""
@group(0) @binding(0) var<storage, read> parameter: array<f32>;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let gid = global_id.x + group_id.y * {row_width}u;
    if (gid < {total}u) {{
        output[gid] = parameter[gid] - {float(learning_rate)!r} * gradient[gid];
    }}
}}
"""
            self._dispatch(
                encoder,
                ("sgd", "copy", total, float(learning_rate)),
                source,
                (parameter, gradient, output),
                ("read", "read", "write"),
                workgroups,
            )
        return output
