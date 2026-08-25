from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .lazy import Compiler, CompiledProgram, DeviceAdapter, LazyNode, Op


class _OpenCLArrayView:
    """Cheap shape/stride view used between generated OpenCL operations."""

    __slots__ = (
        "_base",
        "data",
        "dtype",
        "ndim",
        "offset",
        "shape",
        "size",
        "strides",
        "_to_contiguous",
    )

    def __init__(
        self,
        value: Any,
        shape: Tuple[int, ...],
        strides: Tuple[int, ...],
        to_contiguous: Callable[[Any], Any],
    ) -> None:
        self._base = value._base if isinstance(value, _OpenCLArrayView) else value
        self.data = value.data
        self.dtype = value.dtype
        self.offset = value.offset
        self.shape = shape
        self.strides = strides
        self._to_contiguous = to_contiguous
        self.ndim = len(shape)
        size = 1
        for dimension in shape:
            size *= dimension
        self.size = size

    def add_event(self, event: Any) -> None:
        self._base.add_event(event)

    def get(self) -> np.ndarray:
        expected = np.dtype(self.dtype).itemsize
        for size, stride in zip(reversed(self.shape), reversed(self.strides)):
            if size == 0:
                break
            if size != 1 and stride != expected:
                return self._to_contiguous(self).get()
            expected *= size
        materialized = self._base._new_with_changes(
            data=self.data,
            offset=self.offset,
            shape=self.shape,
            strides=self.strides,
        )
        return materialized.get()


class OpenCLDeviceAdapter(DeviceAdapter):
    name = "opencl"

    def __init__(
        self,
        cl: Any,
        cla: Any,
        context: Any,
        queue: Any,
        allocator: Any,
    ) -> None:
        self._cl = cl
        self._cla = cla
        self.context = context
        self.queue = queue
        self.allocator = allocator

    def array(self, value: object) -> Any:
        return self._cla.to_device(
            self.queue, np.asarray(value), allocator=self.allocator
        )

    def evaluate(self, values: Sequence[object]) -> None:
        self.synchronize()

    def synchronize(self) -> None:
        self.queue.finish()

    def to_numpy(self, value: object) -> np.ndarray:
        return np.asarray(value.get())  # type: ignore[attr-defined]

    def argmax(self, value: object) -> int:
        return int(self.to_numpy(value).argmax())

    def argmax_last_axis(self, value: object) -> np.ndarray:
        return self.to_numpy(value).argmax(axis=-1)

    def dtype(self, value: object) -> np.dtype:
        return np.dtype(value.dtype)  # type: ignore[attr-defined]


@dataclass
class OpenCLProgram(CompiledProgram):
    source: str
    inputs: Tuple[LazyNode, ...]
    _run: Callable[[Sequence[Any]], Tuple[Any, ...]]
    device: OpenCLDeviceAdapter
    _load_input: Callable[[LazyNode], Any]

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
        outputs = tuple(self._run(values))
        if synchronize:
            self.device.evaluate(outputs)
        return outputs


class OpenCLCompiler(Compiler):
    """Execute the portable tensor IR on OpenCL with CLBlast FP32 GEMM."""

    _SUPPORTED_LOWERINGS = {
        "elementwise_fusion",
        "layer_norm",
        "layer_norm_backward",
        "log_softmax",
        "log_softmax_backward",
        "softmax",
        "softmax_backward",
    }
    _LAYOUT_ROW_MAJOR = 101
    _TRANSPOSE_NO = 111
    _TRANSPOSE_YES = 112
    _SGD_FUSION_GROUP_SIZE = 16

    _BINARY_EXPRESSIONS = {
        "add": "left_value + right_value",
        "sub": "left_value - right_value",
        "mul": "left_value * right_value",
        "div": "left_value / right_value",
        "equal": "left_value == right_value ? 1.0f : 0.0f",
    }
    _UNARY_EXPRESSIONS = {
        "identity": "input_value",
        "neg": "-input_value",
        "exp": "exp(input_value)",
        "log": "log(input_value)",
        "sqrt": "sqrt(input_value)",
        "relu": "fmax(input_value, 0.0f)",
        "step": "input_value > 0.0f ? 1.0f : 0.0f",
    }

    def __init__(self) -> None:
        try:
            import pyopencl as cl
            import pyopencl.array as cla
            from pyopencl.tools import ImmediateAllocator, MemoryPool
        except ImportError as error:
            raise ImportError(
                "the OpenCL backend requires `python -m pip install "
                "'chomik-grad[opencl]'`"
            ) from error

        try:
            platforms = cl.get_platforms()
            devices = [
                device
                for platform in platforms
                for device in platform.get_devices()
                if device.type & cl.device_type.GPU
            ]
            if not devices:
                devices = [
                    device
                    for platform in platforms
                    for device in platform.get_devices()
                ]
        except Exception as error:
            raise RuntimeError("the OpenCL backend requires a working ICD") from error
        if not devices:
            raise RuntimeError("the OpenCL backend requires an available device")

        self._cl = cl
        self._cla = cla
        self.context = cl.Context([devices[0]])
        self.queue = cl.CommandQueue(self.context)
        self._allocator = MemoryPool(ImmediateAllocator(self.queue))
        self.device = OpenCLDeviceAdapter(
            cl,
            cla,
            self.context,
            self.queue,
            self._allocator,
        )
        self.device_name = str(devices[0].name).strip()
        self._clblast = self._load_clblast()
        self._configure_clblast()
        self._program_cache: Dict[object, Tuple[str, Callable[..., Any]]] = {}
        self._kernel_cache: Dict[object, Any] = {}
        self._matmul_plan_cache: Dict[object, Tuple[object, ...]] = {}

    @property
    def cache_size(self) -> int:
        return len(self._program_cache)

    def close(self) -> None:
        self.device.synchronize()
        self._program_cache.clear()
        self._kernel_cache.clear()
        self._matmul_plan_cache.clear()
        self._allocator.stop_holding()

    def _load_clblast(self) -> Any:
        configured = os.environ.get("CLBLAST_PATH")
        candidates = []
        if configured:
            path = Path(configured)
            if path.is_dir():
                name = "clblast.dll" if os.name == "nt" else "libclblast.so"
                path = path / name
            candidates.append(str(path))
        discovered = ctypes.util.find_library("clblast")
        if discovered:
            candidates.append(discovered)
        if os.name == "nt":
            candidates.append(
                str(Path(sys.prefix) / "Library" / "bin" / "clblast.dll")
            )
        else:
            candidates.append(
                str(Path(sys.prefix) / "lib" / "libclblast.so")
            )

        errors = []
        for candidate in candidates:
            try:
                return ctypes.CDLL(candidate)
            except OSError as error:
                errors.append(f"{candidate}: {error}")
        detail = f" ({'; '.join(errors)})" if errors else ""
        raise ImportError(
            "the OpenCL backend requires CLBlast 1.7; install the shared "
            "library or set CLBLAST_PATH to clblast.dll/libclblast.so" + detail
        )

    def _configure_clblast(self) -> None:
        pointer = ctypes.c_void_p
        size = ctypes.c_size_t
        function = self._clblast.CLBlastSgemm
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            size,
            size,
            size,
            ctypes.c_float,
            pointer,
            size,
            size,
            pointer,
            size,
            size,
            ctypes.c_float,
            pointer,
            size,
            size,
            ctypes.POINTER(pointer),
            ctypes.POINTER(pointer),
        ]
        function.restype = ctypes.c_int
        self._sgemm = function

        batched = self._clblast.CLBlastSgemmBatched
        batched.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            size,
            size,
            size,
            ctypes.POINTER(ctypes.c_float),
            pointer,
            ctypes.POINTER(size),
            size,
            pointer,
            ctypes.POINTER(size),
            size,
            ctypes.POINTER(ctypes.c_float),
            pointer,
            ctypes.POINTER(size),
            size,
            size,
            ctypes.POINTER(pointer),
            ctypes.POINTER(pointer),
        ]
        batched.restype = ctypes.c_int
        self._sgemm_batched = batched

        strided = self._clblast.CLBlastSgemmStridedBatched
        strided.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            size,
            size,
            size,
            ctypes.c_float,
            pointer,
            size,
            size,
            size,
            pointer,
            size,
            size,
            size,
            ctypes.c_float,
            pointer,
            size,
            size,
            size,
            size,
            ctypes.POINTER(pointer),
            ctypes.POINTER(pointer),
        ]
        strided.restype = ctypes.c_int
        self._sgemm_strided_batched = strided

    def _load_input(self, node: LazyNode) -> Any:
        if "opencl" in node.native_values:
            return node.native_values["opencl"]
        value = node.numpy_value()
        if value.dtype not in (np.float32, np.int32, np.int64):
            raise TypeError(
                "the OpenCL FP32 backend supports float32 tensors and "
                "int32/int64 gather indices"
            )
        native = self.device.array(value)
        if node.cache_native:
            node.native_values["opencl"] = native
        return native

    @classmethod
    def _node_inputs(cls, node: LazyNode) -> Tuple[LazyNode, ...]:
        if (
            node.lowering is not None
            and node.lowering[0] in cls._SUPPORTED_LOWERINGS
        ):
            return node.lowering[1]
        return node.inputs

    @classmethod
    def _topological_sort(
        cls, outputs: Sequence[LazyNode]
    ) -> Tuple[LazyNode, ...]:
        ordered = []
        visited = set()

        def visit(node: LazyNode) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in cls._node_inputs(node):
                visit(parent)
            ordered.append(node)

        for output in outputs:
            visit(output)
        return tuple(ordered)

    @classmethod
    def _signature(
        cls, nodes: Sequence[LazyNode], outputs: Sequence[LazyNode]
    ) -> object:
        indexes = {node: index for index, node in enumerate(nodes)}
        specifications = []
        for node in nodes:
            if node.op is None:
                specifications.append((None, node.shape, node.dtype.str))
            elif (
                node.lowering is not None
                and node.lowering[0] in cls._SUPPORTED_LOWERINGS
            ):
                kind, inputs, argument = node.lowering
                specifications.append(
                    (
                        "lowering",
                        kind,
                        argument,
                        node.shape,
                        node.dtype.str,
                        tuple(indexes[parent] for parent in inputs),
                    )
                )
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

    @classmethod
    def _mark_elementwise_fusions(
        cls, outputs: Sequence[LazyNode]
    ) -> None:
        nodes = cls._topological_sort(outputs)
        consumers: Dict[LazyNode, int] = {}
        for node in nodes:
            for parent in cls._node_inputs(node):
                consumers[parent] = consumers.get(parent, 0) + 1
        output_nodes = set(outputs)

        def eligible(node: LazyNode) -> bool:
            return (
                node.op is Op.ELEMENTWISE
                and node.lowering is None
                and node.dtype == np.dtype(np.float32)
            )

        inlineable = set()
        for child in nodes:
            if not eligible(child):
                continue
            for parent in child.inputs:
                if (
                    eligible(parent)
                    and parent.shape == child.shape
                    and consumers.get(parent) == 1
                    and parent not in output_nodes
                ):
                    inlineable.add(parent)

        for root in nodes:
            if not eligible(root) or root in inlineable:
                continue
            inputs = []
            input_indexes: Dict[LazyNode, int] = {}
            fused_count = 0

            def build(node: LazyNode) -> object:
                nonlocal fused_count
                if node is root or node in inlineable:
                    fused_count += 1
                    return (
                        str(node.arg),
                        *(build(parent) for parent in node.inputs),
                    )
                index = input_indexes.get(node)
                if index is None:
                    index = len(inputs)
                    input_indexes[node] = index
                    inputs.append(node)
                return ("input", index)

            expression = build(root)
            if fused_count > 1:
                root.lowering = (
                    "elementwise_fusion",
                    tuple(inputs),
                    expression,
                )

    def compile(
        self,
        outputs: Sequence[LazyNode],
        dynamic_inputs: Sequence[LazyNode] = (),
    ) -> OpenCLProgram:
        if not outputs:
            raise ValueError("at least one output is required")
        if any(node.op is not None for node in dynamic_inputs):
            raise ValueError("dynamic inputs must be graph leaves")

        nodes = self._topological_sort(outputs)
        leaves = tuple(node for node in nodes if node.op is None)
        requested_dynamic = set(dynamic_inputs)
        if not requested_dynamic.issubset(leaves):
            raise ValueError("dynamic input is not a leaf of the compiled graph")
        specialized = bool(dynamic_inputs)
        signature = self._signature(nodes, outputs)
        cached = None if specialized else self._program_cache.get(signature)
        if cached is not None:
            source, run = cached
            return OpenCLProgram(source, leaves, run, self.device, self._load_input)

        self._mark_elementwise_fusions(outputs)
        nodes = self._topological_sort(outputs)
        leaves = tuple(node for node in nodes if node.op is None)
        program_inputs = tuple(node for node in leaves if node in requested_dynamic)

        names = {node: f"v{index}" for index, node in enumerate(nodes)}
        leaf_indexes = {node: index for index, node in enumerate(leaves)}
        last_uses: Dict[LazyNode, int] = {}
        bundle_last_uses: Dict[object, int] = {}
        for index, node in enumerate(nodes):
            for parent in self._node_inputs(node):
                last_uses[parent] = index
            if (
                node.lowering is not None
                and node.lowering[0] == "layer_norm_backward"
            ):
                _, lowering_inputs, argument = node.lowering
                epsilon, _ = argument
                bundle_last_uses[(lowering_inputs, float(epsilon))] = index
        for output in outputs:
            last_uses[output] = len(nodes)
        releases: Dict[int, list[str]] = {}
        for node, index in last_uses.items():
            if index < len(nodes):
                releases.setdefault(index, []).append(names[node])

        lines = ["def run(inputs):"]
        lowering_bundles: Dict[object, str] = {}
        for index, node in enumerate(nodes):
            name = names[node]
            if node.op is None:
                lines.append(f"    {name} = inputs[{leaf_indexes[node]}]")
                continue
            args = [names[parent] for parent in self._node_inputs(node)]
            if (
                node.lowering is not None
                and node.lowering[0] == "layer_norm_backward"
            ):
                _, lowering_inputs, argument = node.lowering
                epsilon, component = argument
                key = (lowering_inputs, float(epsilon))
                bundle = lowering_bundles.get(key)
                if bundle is None:
                    bundle = f"{name}_bundle"
                    lowering_bundles[key] = bundle
                    lines.append(
                        f"    {bundle} = layer_norm_backward("
                        f"{args[0]}, {args[1]}, {args[2]}, "
                        f"{float(epsilon)!r})"
                    )
                lines.append(f"    {name} = {bundle}[{int(component)}]")
                if bundle_last_uses[key] == index:
                    lines.append(f"    del {bundle}")
            else:
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
            compile(source, "<chomikgrad-opencl>", "exec"),
            {
                "binary": self._binary,
                "fused_elementwise": self._fused_elementwise,
                "gather": self._gather,
                "layer_norm": self._layer_norm,
                "layer_norm_backward": self._layer_norm_backward,
                "log_softmax_backward": self._log_softmax_backward,
                "matmul": self._matmul,
                "permute": self._permute,
                "reduce": self._reduce,
                "reshape": self._reshape,
                "softmax": self._softmax,
                "softmax_backward": self._softmax_backward,
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

            def run(values: Sequence[Any]) -> Tuple[Any, ...]:
                merged = list(template)
                for leaf_index, node in enumerate(leaves):
                    if node in dynamic_indexes:
                        merged[leaf_index] = values[dynamic_indexes[node]]
                return raw_run(merged)

        else:
            run = raw_run
            self._program_cache[signature] = source, run

        return OpenCLProgram(  # type: ignore[arg-type]
            source,
            program_inputs if specialized else leaves,
            run,
            self.device,
            self._load_input,
        )

    def _expression(self, node: LazyNode, args: Sequence[str]) -> str:
        if node.dtype != np.dtype(np.float32) and node.op is not Op.GATHER:
            raise TypeError("the OpenCL backend currently supports FP32 operations")
        if (
            node.lowering is not None
            and node.lowering[0] in self._SUPPORTED_LOWERINGS
        ):
            kind, _, argument = node.lowering
            if kind == "elementwise_fusion":
                rendered_inputs = ", ".join(args)
                if len(args) == 1:
                    rendered_inputs += ","
                return (
                    f"fused_elementwise({argument!r}, "
                    f"({rendered_inputs}), {node.shape!r})"
                )
            if kind == "layer_norm":
                return (
                    f"layer_norm({args[0]}, {args[1]}, {args[2]}, "
                    f"{float(argument)!r})"
                )
            if kind == "softmax":
                return f"softmax({args[0]}, {int(argument)!r}, False)"
            if kind == "log_softmax":
                return f"softmax({args[0]}, {int(argument)!r}, True)"
            if kind == "softmax_backward":
                return (
                    f"softmax_backward({args[0]}, {args[1]}, "
                    f"{int(argument)!r})"
                )
            if kind == "log_softmax_backward":
                return (
                    f"log_softmax_backward({args[0]}, {args[1]}, "
                    f"{int(argument)!r})"
                )
            if kind == "layer_norm_backward":
                raise RuntimeError(
                    "layer_norm_backward must be emitted as a shared bundle"
                )
            raise ValueError(f"unsupported OpenCL lowering: {kind}")
        if node.op is Op.ELEMENTWISE:
            kind = str(node.arg)
            if kind in self._BINARY_EXPRESSIONS:
                return f"binary({kind!r}, {args[0]}, {args[1]}, {node.shape!r})"
            if kind in self._UNARY_EXPRESSIONS:
                return f"unary({kind!r}, {args[0]}, {node.shape!r})"
            raise ValueError(f"unsupported elementwise operation: {kind}")
        if node.op is Op.REDUCE:
            kind, axes, keepdims = node.arg  # type: ignore[misc]
            return (
                f"reduce({kind!r}, {args[0]}, {tuple(axes)!r}, "
                f"{bool(keepdims)!r}, {node.shape!r})"
            )
        if node.op is Op.RESHAPE:
            return f"reshape({args[0]}, {node.shape!r})"
        if node.op is Op.PERMUTE:
            return f"permute({args[0]}, {tuple(node.arg)!r})"
        if node.op is Op.MATMUL:
            return f"matmul({args[0]}, {args[1]}, {node.shape!r})"
        if node.op is Op.GATHER:
            return f"gather({args[0]}, {args[1]}, {node.shape!r})"
        raise ValueError(f"unsupported operation: {node.op}")

    @staticmethod
    def _strides_in_elements(value: Any) -> Tuple[int, ...]:
        itemsize = np.dtype(value.dtype).itemsize
        return tuple(int(stride // itemsize) for stride in value.strides)

    @staticmethod
    def _c_strides(shape: Sequence[int]) -> Tuple[int, ...]:
        strides = []
        running = 1
        for size in reversed(shape):
            strides.append(running)
            running *= int(size)
        return tuple(reversed(strides))

    @classmethod
    def _c_byte_strides(
        cls, shape: Sequence[int], dtype: np.dtype
    ) -> Tuple[int, ...]:
        itemsize = np.dtype(dtype).itemsize
        return tuple(stride * itemsize for stride in cls._c_strides(shape))

    @staticmethod
    def _is_c_contiguous(value: Any) -> bool:
        expected = np.dtype(value.dtype).itemsize
        for size, stride in zip(reversed(value.shape), reversed(value.strides)):
            if size == 0:
                return True
            if size != 1 and stride != expected:
                return False
            expected *= size
        return True

    @classmethod
    def _broadcast_offset_expression(
        cls,
        value: Any,
        output_shape: Tuple[int, ...],
        index_name: str,
        offset_name: str,
    ) -> str:
        rank = len(output_shape)
        padding = rank - value.ndim
        shapes = (1,) * padding + tuple(value.shape)
        strides = (0,) * padding + cls._strides_in_elements(value)
        output_strides = cls._c_strides(output_shape)
        terms = [offset_name]
        for axis, (size, stride) in enumerate(zip(shapes, strides)):
            if size != 1 and stride:
                terms.append(
                    f"((({index_name} / {output_strides[axis]}) % "
                    f"{output_shape[axis]}) * {stride})"
                )
        return " + ".join(terms)

    def _empty(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> Any:
        return self._cla.empty(
            self.queue, shape, dtype, allocator=self._allocator
        )

    def _build_kernel(
        self,
        key: object,
        source: str,
        name: str,
        scalar_arg_dtypes: Sequence[object],
    ) -> Any:
        cached = self._kernel_cache.get(key)
        if cached is not None:
            return cached
        kernel = getattr(self._cl.Program(self.context, source).build(), name)
        kernel.set_scalar_arg_dtypes(scalar_arg_dtypes)
        self._kernel_cache[key] = kernel
        return kernel

    @staticmethod
    def _record_event(event: Any, *values: Any) -> None:
        for value in values:
            value.add_event(event)

    def _fused_elementwise(
        self,
        expression: object,
        inputs: Sequence[Any],
        output_shape: Tuple[int, ...],
    ) -> Any:
        if any(value.dtype != np.float32 for value in inputs):
            raise TypeError("OpenCL elementwise operations require float32")
        total = int(np.prod(output_shape, dtype=np.int64))
        output = self._empty(output_shape)
        if total == 0:
            return output

        def render(node: object) -> str:
            parts = tuple(node)  # type: ignore[arg-type]
            kind = str(parts[0])
            if kind == "input":
                return f"input_{int(parts[1])}_value"
            if kind in self._BINARY_EXPRESSIONS:
                left = render(parts[1])
                right = render(parts[2])
                return (
                    self._BINARY_EXPRESSIONS[kind]
                    .replace("left_value", f"({left})")
                    .replace("right_value", f"({right})")
                )
            if kind in self._UNARY_EXPRESSIONS:
                value = render(parts[1])
                return self._UNARY_EXPRESSIONS[kind].replace(
                    "input_value", f"({value})"
                )
            raise ValueError(f"unsupported fused elementwise operation: {kind}")

        arguments = []
        loads = []
        scalar_types = []
        runtime_arguments = []
        key_inputs = []
        for index, value in enumerate(inputs):
            offset_name = f"input_{index}_offset"
            offset = self._broadcast_offset_expression(
                value, output_shape, "gid", offset_name
            )
            arguments.extend(
                (
                    f"    __global const float* input_{index}",
                    f"    const ulong {offset_name}",
                )
            )
            loads.append(
                f"    const float input_{index}_value = input_{index}[{offset}];"
            )
            scalar_types.extend((None, np.uint64))
            runtime_arguments.extend(
                (
                    value.data,
                    np.uint64(value.offset // value.dtype.itemsize),
                )
            )
            key_inputs.append((tuple(value.shape), tuple(value.strides)))
        rendered_arguments = ",\n".join(arguments)
        if rendered_arguments:
            rendered_arguments += ",\n"
        rendered_loads = "\n".join(loads)
        rendered_expression = render(expression)
        source = f"""
__kernel void fused_elementwise_op(
{rendered_arguments}    __global float* output,
    const ulong total
) {{
    const ulong gid = get_global_id(0);
    if (gid >= total) return;
{rendered_loads}
    output[gid] = {rendered_expression};
}}
"""
        key = (
            "fused_elementwise",
            expression,
            tuple(key_inputs),
            output_shape,
        )
        kernel = self._build_kernel(
            key,
            source,
            "fused_elementwise_op",
            (*scalar_types, None, np.uint64),
        )
        event = kernel(
            self.queue,
            (total,),
            None,
            *runtime_arguments,
            output.data,
            np.uint64(total),
        )
        self._record_event(event, *inputs, output)
        return output

    def _binary(
        self,
        kind: str,
        left: Any,
        right: Any,
        output_shape: Tuple[int, ...],
    ) -> Any:
        if left.dtype != np.float32 or right.dtype != np.float32:
            raise TypeError("OpenCL elementwise operations require float32")
        total = int(np.prod(output_shape, dtype=np.int64))
        output = self._empty(output_shape)
        if total == 0:
            return output
        left_offset = self._broadcast_offset_expression(
            left, output_shape, "gid", "left_offset"
        )
        right_offset = self._broadcast_offset_expression(
            right, output_shape, "gid", "right_offset"
        )
        expression = self._BINARY_EXPRESSIONS[kind]
        source = f"""
__kernel void binary_op(
    __global const float* left,
    const ulong left_offset,
    __global const float* right,
    const ulong right_offset,
    __global float* output,
    const ulong total
) {{
    const ulong gid = get_global_id(0);
    if (gid >= total) return;
    const float left_value = left[{left_offset}];
    const float right_value = right[{right_offset}];
    output[gid] = {expression};
}}
"""
        key = (
            "binary",
            kind,
            tuple(left.shape),
            tuple(left.strides),
            tuple(right.shape),
            tuple(right.strides),
            output_shape,
        )
        kernel = self._build_kernel(
            key,
            source,
            "binary_op",
            [None, np.uint64, None, np.uint64, None, np.uint64],
        )
        event = kernel(
            self.queue,
            (total,),
            None,
            left.data,
            np.uint64(left.offset // left.dtype.itemsize),
            right.data,
            np.uint64(right.offset // right.dtype.itemsize),
            output.data,
            np.uint64(total),
        )
        self._record_event(event, left, right, output)
        return output

    def _unary(
        self,
        kind: str,
        value: Any,
        output_shape: Tuple[int, ...],
    ) -> Any:
        if value.dtype != np.float32:
            raise TypeError("OpenCL elementwise operations require float32")
        total = int(np.prod(output_shape, dtype=np.int64))
        output = self._empty(output_shape)
        if total == 0:
            return output
        input_offset = self._broadcast_offset_expression(
            value, output_shape, "gid", "input_offset"
        )
        expression = self._UNARY_EXPRESSIONS[kind]
        source = f"""
__kernel void unary_op(
    __global const float* input,
    const ulong input_offset,
    __global float* output,
    const ulong total
) {{
    const ulong gid = get_global_id(0);
    if (gid >= total) return;
    const float input_value = input[{input_offset}];
    output[gid] = {expression};
}}
"""
        key = (
            "unary",
            kind,
            tuple(value.shape),
            tuple(value.strides),
            output_shape,
        )
        kernel = self._build_kernel(
            key,
            source,
            "unary_op",
            [None, np.uint64, None, np.uint64],
        )
        event = kernel(
            self.queue,
            (total,),
            None,
            value.data,
            np.uint64(value.offset // value.dtype.itemsize),
            output.data,
            np.uint64(total),
        )
        self._record_event(event, value, output)
        return output

    def _reduce(
        self,
        kind: str,
        value: Any,
        axes: Tuple[int, ...],
        keepdims: bool,
        output_shape: Tuple[int, ...],
    ) -> Any:
        del keepdims
        if value.dtype != np.float32:
            raise TypeError("OpenCL reductions require float32")
        retained = tuple(axis for axis in range(value.ndim) if axis not in axes)
        reduced_shapes = tuple(value.shape[axis] for axis in axes)
        retained_shapes = tuple(value.shape[axis] for axis in retained)
        reduced_strides = self._c_strides(reduced_shapes)
        retained_strides = self._c_strides(retained_shapes)
        input_strides = self._strides_in_elements(value)
        terms = ["input_offset"]
        for index, axis in enumerate(retained):
            terms.append(
                f"(((output_index / {retained_strides[index]}) % "
                f"{retained_shapes[index]}) * {input_strides[axis]})"
            )
        for index, axis in enumerate(axes):
            terms.append(
                f"(((reduction_index / {reduced_strides[index]}) % "
                f"{reduced_shapes[index]}) * {input_strides[axis]})"
            )
        source_offset = " + ".join(terms)
        neutral = "0.0f" if kind == "sum" else "-INFINITY"
        combine = "accumulator += current" if kind == "sum" else (
            "accumulator = fmax(accumulator, current)"
        )
        merge = "scratch[lane] += scratch[lane + stride]" if kind == "sum" else (
            "scratch[lane] = fmax(scratch[lane], scratch[lane + stride])"
        )
        reduction_count = int(np.prod(reduced_shapes, dtype=np.int64))
        output_count = int(np.prod(output_shape, dtype=np.int64))
        output = self._empty(output_shape)
        if output_count == 0:
            return output
        source = f"""
__kernel void reduce_op(
    __global const float* input,
    const ulong input_offset,
    __global float* output
) {{
    const ulong output_index = get_group_id(0);
    const ulong lane = get_local_id(0);
    __local float scratch[256];
    float accumulator = {neutral};
    for (ulong reduction_index = lane;
         reduction_index < {reduction_count};
         reduction_index += 256) {{
        const float current = input[{source_offset}];
        {combine};
    }}
    scratch[lane] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (ulong stride = 128; stride > 0; stride /= 2) {{
        if (lane < stride) {merge};
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    if (lane == 0) output[output_index] = scratch[0];
}}
"""
        key = (
            "reduce",
            kind,
            tuple(value.shape),
            tuple(value.strides),
            axes,
            output_shape,
        )
        kernel = self._build_kernel(
            key,
            source,
            "reduce_op",
            [None, np.uint64, None],
        )
        event = kernel(
            self.queue,
            (output_count * 256,),
            (256,),
            value.data,
            np.uint64(value.offset // value.dtype.itemsize),
            output.data,
        )
        self._record_event(event, value, output)
        return output

    def _contiguous(self, value: Any) -> Any:
        if self._is_c_contiguous(value):
            return value
        return self._unary("identity", value, tuple(value.shape))

    def _reshape(self, value: Any, shape: Tuple[int, ...]) -> Any:
        contiguous = self._contiguous(value)
        return _OpenCLArrayView(
            contiguous,
            shape,
            self._c_byte_strides(shape, contiguous.dtype),
            self._contiguous,
        )

    def _permute(self, value: Any, axes: Tuple[int, ...]) -> Any:
        return _OpenCLArrayView(
            value,
            tuple(value.shape[axis] for axis in axes),
            tuple(value.strides[axis] for axis in axes),
            self._contiguous,
        )

    def _layer_norm(
        self,
        inputs: Any,
        weight: Any,
        bias: Any,
        epsilon: float,
    ) -> Any:
        inputs = self._contiguous(inputs)
        weight = self._contiguous(weight)
        bias = self._contiguous(bias)
        if (
            inputs.dtype != np.float32
            or weight.dtype != np.float32
            or bias.dtype != np.float32
        ):
            raise TypeError("OpenCL layer norm requires float32")
        output = self._empty(tuple(inputs.shape))
        if inputs.size == 0:
            return output
        width = int(inputs.shape[-1])
        rows = int(inputs.size // width)
        source = r"""
__kernel void layer_norm(
    __global const float* inputs,
    __global const float* weight,
    __global const float* bias,
    __global float* outputs,
    const int width,
    const float epsilon
) {
    const int row = get_group_id(0);
    const int lane = get_local_id(0);
    const int offset = row * width;
    __local float values[256];

    float sum = 0.0f;
    for (int column = lane; column < width; column += 256) {
        sum += inputs[offset + column];
    }
    values[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) values[lane] += values[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mean = values[0] / width;
    barrier(CLK_LOCAL_MEM_FENCE);

    float squared_sum = 0.0f;
    for (int column = lane; column < width; column += 256) {
        const float centered = inputs[offset + column] - mean;
        squared_sum += centered * centered;
    }
    values[lane] = squared_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) values[lane] += values[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inverse_std = rsqrt(values[0] / width + epsilon);

    for (int column = lane; column < width; column += 256) {
        outputs[offset + column] =
            (inputs[offset + column] - mean) * inverse_std * weight[column]
            + bias[column];
    }
}
"""
        kernel = self._build_kernel(
            "layer_norm",
            source,
            "layer_norm",
            [None, None, None, None, np.int32, np.float32],
        )
        event = kernel(
            self.queue,
            (rows * 256,),
            (256,),
            inputs.data,
            weight.data,
            bias.data,
            output.data,
            np.int32(width),
            np.float32(epsilon),
        )
        self._record_event(event, inputs, weight, bias, output)
        return output

    def _softmax(self, inputs: Any, axis: int, logarithmic: bool) -> Any:
        if axis != inputs.ndim - 1:
            reduction_shape = tuple(
                1 if index == axis else size
                for index, size in enumerate(inputs.shape)
            )
            maximum = self._reduce(
                "max", inputs, (axis,), True, reduction_shape
            )
            shifted = self._binary(
                "sub", inputs, maximum, tuple(inputs.shape)
            )
            exponentials = self._unary("exp", shifted, tuple(inputs.shape))
            total = self._reduce(
                "sum", exponentials, (axis,), True, reduction_shape
            )
            if logarithmic:
                normalizer = self._unary("log", total, reduction_shape)
                return self._binary(
                    "sub", shifted, normalizer, tuple(inputs.shape)
                )
            return self._binary(
                "div", exponentials, total, tuple(inputs.shape)
            )

        inputs = self._contiguous(inputs)
        if inputs.dtype != np.float32:
            raise TypeError("OpenCL softmax requires float32")
        output = self._empty(tuple(inputs.shape))
        if inputs.size == 0:
            return output
        width = int(inputs.shape[-1])
        rows = int(inputs.size // width)
        source = r"""
__kernel void softmax_forward(
    __global const float* inputs,
    __global float* output,
    const int width,
    const int logarithmic
) {
    const int row = get_group_id(0);
    const int lane = get_local_id(0);
    const int offset = row * width;
    __local float values[256];

    float maximum = -INFINITY;
    for (int column = lane; column < width; column += 256) {
        maximum = fmax(maximum, inputs[offset + column]);
    }
    values[lane] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) values[lane] = fmax(values[lane], values[lane + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum = values[0];

    float total = 0.0f;
    for (int column = lane; column < width; column += 256) {
        total += exp(inputs[offset + column] - maximum);
    }
    values[lane] = total;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) values[lane] += values[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    total = values[0];

    for (int column = lane; column < width; column += 256) {
        const int index = offset + column;
        const float shifted = inputs[index] - maximum;
        output[index] = logarithmic ? shifted - log(total) : exp(shifted) / total;
    }
}
"""
        kernel = self._build_kernel(
            "softmax_forward",
            source,
            "softmax_forward",
            [None, None, np.int32, np.int32],
        )
        event = kernel(
            self.queue,
            (rows * 256,),
            (256,),
            inputs.data,
            output.data,
            np.int32(width),
            np.int32(logarithmic),
        )
        self._record_event(event, inputs, output)
        return output

    def _log_softmax_backward(
        self,
        gradients: Any,
        outputs: Any,
        axis: int,
    ) -> Any:
        if axis != gradients.ndim - 1:
            projection_shape = tuple(
                1 if index == axis else size
                for index, size in enumerate(gradients.shape)
            )
            projection = self._reduce(
                "sum", gradients, (axis,), True, projection_shape
            )
            probabilities = self._unary(
                "exp", outputs, tuple(outputs.shape)
            )
            correction = self._binary(
                "mul", probabilities, projection, tuple(outputs.shape)
            )
            return self._binary(
                "sub", gradients, correction, tuple(gradients.shape)
            )

        gradients = self._contiguous(gradients)
        outputs = self._contiguous(outputs)
        if gradients.dtype != np.float32 or outputs.dtype != np.float32:
            raise TypeError("OpenCL log-softmax backward requires float32")
        input_gradients = self._empty(tuple(gradients.shape))
        if gradients.size == 0:
            return input_gradients
        width = int(gradients.shape[-1])
        rows = int(gradients.size // width)
        source = r"""
__kernel void log_softmax_backward(
    __global const float* gradients,
    __global const float* outputs,
    __global float* input_gradients,
    const int width
) {
    const int row = get_group_id(0);
    const int lane = get_local_id(0);
    const int offset = row * width;
    __local float values[256];

    float projection = 0.0f;
    for (int column = lane; column < width; column += 256) {
        projection += gradients[offset + column];
    }
    values[lane] = projection;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) values[lane] += values[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    projection = values[0];

    for (int column = lane; column < width; column += 256) {
        const int index = offset + column;
        input_gradients[index] =
            gradients[index] - exp(outputs[index]) * projection;
    }
}
"""
        kernel = self._build_kernel(
            "log_softmax_backward",
            source,
            "log_softmax_backward",
            [None, None, None, np.int32],
        )
        event = kernel(
            self.queue,
            (rows * 256,),
            (256,),
            gradients.data,
            outputs.data,
            input_gradients.data,
            np.int32(width),
        )
        self._record_event(event, gradients, outputs, input_gradients)
        return input_gradients

    def _softmax_backward(
        self,
        gradients: Any,
        outputs: Any,
        axis: int,
    ) -> Any:
        if axis != gradients.ndim - 1:
            weighted = self._binary(
                "mul", gradients, outputs, tuple(gradients.shape)
            )
            projection_shape = tuple(
                1 if index == axis else size
                for index, size in enumerate(gradients.shape)
            )
            projection = self._reduce(
                "sum", weighted, (axis,), True, projection_shape
            )
            difference = self._binary(
                "sub", gradients, projection, tuple(gradients.shape)
            )
            return self._binary(
                "mul", outputs, difference, tuple(gradients.shape)
            )

        gradients = self._contiguous(gradients)
        outputs = self._contiguous(outputs)
        if gradients.dtype != np.float32 or outputs.dtype != np.float32:
            raise TypeError("OpenCL softmax backward requires float32")
        input_gradients = self._empty(tuple(gradients.shape))
        if gradients.size == 0:
            return input_gradients
        width = int(gradients.shape[-1])
        rows = int(gradients.size // width)
        source = r"""
__kernel void softmax_backward(
    __global const float* gradients,
    __global const float* outputs,
    __global float* input_gradients,
    const int width
) {
    const int row = get_group_id(0);
    const int lane = get_local_id(0);
    const int offset = row * width;
    __local float values[256];

    float projection = 0.0f;
    for (int column = lane; column < width; column += 256) {
        projection += gradients[offset + column] * outputs[offset + column];
    }
    values[lane] = projection;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) values[lane] += values[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    projection = values[0];

    for (int column = lane; column < width; column += 256) {
        const int index = offset + column;
        input_gradients[index] =
            outputs[index] * (gradients[index] - projection);
    }
}
"""
        kernel = self._build_kernel(
            "softmax_backward",
            source,
            "softmax_backward",
            [None, None, None, np.int32],
        )
        event = kernel(
            self.queue,
            (rows * 256,),
            (256,),
            gradients.data,
            outputs.data,
            input_gradients.data,
            np.int32(width),
        )
        self._record_event(event, gradients, outputs, input_gradients)
        return input_gradients

    def _layer_norm_backward(
        self,
        gradients: Any,
        inputs: Any,
        weight: Any,
        epsilon: float,
    ) -> Tuple[Any, Any, Any]:
        gradients = self._contiguous(gradients)
        inputs = self._contiguous(inputs)
        weight = self._contiguous(weight)
        if (
            gradients.dtype != np.float32
            or inputs.dtype != np.float32
            or weight.dtype != np.float32
        ):
            raise TypeError("OpenCL layer norm backward requires float32")

        input_gradients = self._empty(tuple(inputs.shape))
        if inputs.size == 0:
            return (
                input_gradients,
                self._cla.zeros(
                    self.queue,
                    tuple(weight.shape),
                    np.float32,
                    allocator=self._allocator,
                ),
                self._cla.zeros(
                    self.queue,
                    tuple(weight.shape),
                    np.float32,
                    allocator=self._allocator,
                ),
            )
        weight_gradients = self._empty(tuple(weight.shape))
        bias_gradients = self._empty(tuple(weight.shape))
        normalized = self._empty(tuple(inputs.shape))
        width = int(inputs.shape[-1])
        rows = int(inputs.size // width)
        input_source = r"""
__kernel void layer_norm_input_backward(
    __global const float* gradients,
    __global const float* inputs,
    __global const float* weight,
    __global float* input_gradients,
    __global float* normalized_values,
    const int width,
    const float epsilon
) {
    const int row = get_group_id(0);
    const int lane = get_local_id(0);
    const int offset = row * width;
    __local float values[256];

    float sum = 0.0f;
    for (int column = lane; column < width; column += 256) {
        sum += inputs[offset + column];
    }
    values[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) values[lane] += values[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mean = values[0] / width;
    barrier(CLK_LOCAL_MEM_FENCE);

    float squared_sum = 0.0f;
    for (int column = lane; column < width; column += 256) {
        const float centered = inputs[offset + column] - mean;
        squared_sum += centered * centered;
    }
    values[lane] = squared_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) values[lane] += values[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float inverse_std = rsqrt(values[0] / width + epsilon);
    barrier(CLK_LOCAL_MEM_FENCE);

    float weighted_sum = 0.0f;
    float correlation_sum = 0.0f;
    for (int column = lane; column < width; column += 256) {
        const int index = offset + column;
        const float normalized = (inputs[index] - mean) * inverse_std;
        const float weighted = gradients[index] * weight[column];
        normalized_values[index] = normalized;
        weighted_sum += weighted;
        correlation_sum += weighted * normalized;
    }
    values[lane] = weighted_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) values[lane] += values[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float weighted_mean = values[0] / width;
    barrier(CLK_LOCAL_MEM_FENCE);

    values[lane] = correlation_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) values[lane] += values[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float correlation_mean = values[0] / width;

    for (int column = lane; column < width; column += 256) {
        const int index = offset + column;
        const float normalized = normalized_values[index];
        const float weighted = gradients[index] * weight[column];
        input_gradients[index] =
            (weighted - weighted_mean - normalized * correlation_mean)
            * inverse_std;
    }
}
"""
        parameter_source = r"""
__kernel void layer_norm_parameter_backward(
    __global const float* gradients,
    __global const float* normalized_values,
    __global float* weight_gradients,
    __global float* bias_gradients,
    const int width,
    const int rows
) {
    const int column = get_group_id(0);
    const int lane = get_local_id(0);
    __local float weight_values[256];
    __local float bias_values[256];
    float weight_sum = 0.0f;
    float bias_sum = 0.0f;
    for (int row = lane; row < rows; row += 256) {
        const int index = row * width + column;
        const float gradient = gradients[index];
        weight_sum += gradient * normalized_values[index];
        bias_sum += gradient;
    }
    weight_values[lane] = weight_sum;
    bias_values[lane] = bias_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) {
            weight_values[lane] += weight_values[lane + stride];
            bias_values[lane] += bias_values[lane + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) {
        weight_gradients[column] = weight_values[0];
        bias_gradients[column] = bias_values[0];
    }
}
"""
        input_kernel = self._build_kernel(
            "layer_norm_input_backward",
            input_source,
            "layer_norm_input_backward",
            [None, None, None, None, None, np.int32, np.float32],
        )
        input_event = input_kernel(
            self.queue,
            (rows * 256,),
            (256,),
            gradients.data,
            inputs.data,
            weight.data,
            input_gradients.data,
            normalized.data,
            np.int32(width),
            np.float32(epsilon),
        )
        self._record_event(
            input_event,
            gradients,
            inputs,
            weight,
            input_gradients,
            normalized,
        )

        parameter_kernel = self._build_kernel(
            "layer_norm_parameter_backward",
            parameter_source,
            "layer_norm_parameter_backward",
            [None, None, None, None, np.int32, np.int32],
        )
        parameter_event = parameter_kernel(
            self.queue,
            (width * 256,),
            (256,),
            gradients.data,
            normalized.data,
            weight_gradients.data,
            bias_gradients.data,
            np.int32(width),
            np.int32(rows),
        )
        self._record_event(
            parameter_event,
            gradients,
            normalized,
            weight_gradients,
            bias_gradients,
        )
        return input_gradients, weight_gradients, bias_gradients

    def _matrix_descriptor(self, value: Any) -> Tuple[Any, int, int]:
        strides = self._strides_in_elements(value)
        if strides[-1] == 1:
            return value, self._TRANSPOSE_NO, strides[-2]
        if strides[-2] == 1:
            return value, self._TRANSPOSE_YES, strides[-1]
        contiguous = self._contiguous(value)
        strides = self._strides_in_elements(contiguous)
        return contiguous, self._TRANSPOSE_NO, strides[-2]

    @classmethod
    def _batch_offset(
        cls,
        value: Any,
        output_batch_shape: Tuple[int, ...],
        coordinates: Tuple[int, ...],
    ) -> int:
        padding = len(output_batch_shape) - (value.ndim - 2)
        shapes = (1,) * padding + tuple(value.shape[:-2])
        strides = (0,) * padding + cls._strides_in_elements(value)[:-2]
        offset = int(value.offset // value.dtype.itemsize)
        for size, stride, coordinate in zip(shapes, strides, coordinates):
            if size != 1:
                offset += coordinate * stride
        return offset

    def _matmul(
        self,
        left: Any,
        right: Any,
        output_shape: Tuple[int, ...],
    ) -> Any:
        if left.dtype != np.float32 or right.dtype != np.float32:
            raise TypeError("OpenCL matmul requires float32")
        left, left_transpose, left_ld = self._matrix_descriptor(left)
        right, right_transpose, right_ld = self._matrix_descriptor(right)
        output = self._empty(output_shape)
        if output.size == 0:
            return output
        batch_shape = output_shape[:-2]
        batch_count = int(np.prod(batch_shape, dtype=np.int64))
        m, n = output_shape[-2:]
        k = left.shape[-1]
        queue_handle = ctypes.c_void_p(self.queue.int_ptr)
        left_base = int(left.offset // left.dtype.itemsize)
        right_base = int(right.offset // right.dtype.itemsize)
        output_base = int(output.offset // output.dtype.itemsize)
        plan_key = (
            tuple(left.shape),
            self._strides_in_elements(left),
            tuple(right.shape),
            self._strides_in_elements(right),
            output_shape,
        )
        plan = self._matmul_plan_cache.get(plan_key)
        if plan is None:
            coordinates = [
                tuple(np.unravel_index(batch, batch_shape))
                if batch_shape
                else ()
                for batch in range(batch_count)
            ]
            left_offsets = tuple(
                self._batch_offset(left, batch_shape, coordinate) - left_base
                for coordinate in coordinates
            )
            right_offsets = tuple(
                self._batch_offset(right, batch_shape, coordinate) - right_base
                for coordinate in coordinates
            )

            def regular_stride(offsets: Sequence[int]) -> Optional[int]:
                if len(offsets) < 2:
                    return 0
                stride = offsets[1] - offsets[0]
                if all(
                    offset == offsets[0] + index * stride
                    for index, offset in enumerate(offsets)
                ):
                    return stride
                return None

            left_stride = regular_stride(left_offsets)
            right_stride = regular_stride(right_offsets)
            alphas = None
            betas = None
            offset_cache = None
            if not (
                batch_count > 1
                and left_stride is not None
                and right_stride is not None
            ):
                float_array = ctypes.c_float * batch_count
                alphas = float_array(*([1.0] * batch_count))
                betas = float_array(*([0.0] * batch_count))
                offset_cache = {}
            plan = (
                left_offsets,
                right_offsets,
                left_stride,
                right_stride,
                alphas,
                betas,
                offset_cache,
            )
            self._matmul_plan_cache[plan_key] = plan
        (
            left_offsets,
            right_offsets,
            left_stride,
            right_stride,
            alphas,
            betas,
            offset_cache,
        ) = plan
        if batch_count > 1 and left_stride is not None and right_stride is not None:
            event_handle = ctypes.c_void_p()
            status = self._sgemm_strided_batched(
                self._LAYOUT_ROW_MAJOR,
                left_transpose,
                right_transpose,
                m,
                n,
                k,
                ctypes.c_float(1.0),
                ctypes.c_void_p(left.data.int_ptr),
                left_base + left_offsets[0],
                left_ld,
                left_stride,
                ctypes.c_void_p(right.data.int_ptr),
                right_base + right_offsets[0],
                right_ld,
                right_stride,
                ctypes.c_float(0.0),
                ctypes.c_void_p(output.data.int_ptr),
                output_base,
                n,
                m * n,
                batch_count,
                ctypes.byref(queue_handle),
                ctypes.byref(event_handle),
            )
            if status != 0:
                raise RuntimeError(
                    f"CLBlast strided-batched SGEMM failed with status {status}"
                )
            event = self._cl.Event.from_int_ptr(
                event_handle.value, retain=False
            )
            self._record_event(event, left, right, output)
            return output

        offset_key = (left_base, right_base, output_base)
        cached_offsets = offset_cache.get(offset_key)
        if cached_offsets is None:
            offset_array = ctypes.c_size_t * batch_count
            cached_offsets = (
                offset_array(
                    *(left_base + offset for offset in left_offsets)
                ),
                offset_array(
                    *(right_base + offset for offset in right_offsets)
                ),
                offset_array(
                    *(output_base + batch * m * n for batch in range(batch_count))
                ),
            )
            offset_cache[offset_key] = cached_offsets
        left_offset_values, right_offset_values, output_offsets = cached_offsets
        event_handle = ctypes.c_void_p()
        status = self._sgemm_batched(
            self._LAYOUT_ROW_MAJOR,
            left_transpose,
            right_transpose,
            m,
            n,
            k,
            alphas,
            ctypes.c_void_p(left.data.int_ptr),
            left_offset_values,
            left_ld,
            ctypes.c_void_p(right.data.int_ptr),
            right_offset_values,
            right_ld,
            betas,
            ctypes.c_void_p(output.data.int_ptr),
            output_offsets,
            n,
            batch_count,
            ctypes.byref(queue_handle),
            ctypes.byref(event_handle),
        )
        if status != 0:
            raise RuntimeError(f"CLBlast batched SGEMM failed with status {status}")
        event = self._cl.Event.from_int_ptr(event_handle.value, retain=False)
        self._record_event(event, left, right, output)
        return output

    def _gather(
        self,
        source: Any,
        indices: Any,
        output_shape: Tuple[int, ...],
    ) -> Any:
        if source.dtype != np.float32:
            raise TypeError("OpenCL gather currently requires float32 values")
        if indices.dtype not in (np.int32, np.int64):
            raise TypeError("OpenCL gather indices must be int32 or int64")
        source = self._contiguous(source)
        indices = self._contiguous(indices)
        output = self._empty(output_shape)
        total = int(np.prod(output_shape, dtype=np.int64))
        if total == 0:
            return output
        width = int(np.prod(source.shape[1:], dtype=np.int64))
        index_type = "int" if indices.dtype == np.int32 else "long"
        source_offset = int(source.offset // source.dtype.itemsize)
        indices_offset = int(indices.offset // indices.dtype.itemsize)
        source_code = f"""
__kernel void gather_op(
    __global const float* source,
    __global const {index_type}* indices,
    __global float* output,
    const ulong total
) {{
    const ulong gid = get_global_id(0);
    if (gid >= total) return;
    const ulong item = gid / {width};
    const ulong column = gid % {width};
    const long row = (long)indices[{indices_offset} + item];
    output[gid] = source[{source_offset} + row * {width} + column];
}}
"""
        key = ("gather", tuple(source.shape), tuple(indices.shape), indices.dtype.str)
        kernel = self._build_kernel(
            key,
            source_code,
            "gather_op",
            [None, None, None, np.uint64],
        )
        event = kernel(
            self.queue,
            (total,),
            None,
            source.data,
            indices.data,
            output.data,
            np.uint64(total),
        )
        self._record_event(event, source, indices, output)
        return output

    def update_parameters(
        self,
        parameters: Sequence[LazyNode],
        gradients: Sequence[LazyNode],
        learning_rate: float,
        *,
        inplace: bool = False,
    ) -> Tuple[Tuple[LazyNode, ...], Tuple[LazyNode, ...]]:
        native_gradients = self.compile(gradients).run(synchronize=False)
        return self.update_native_parameters(
            parameters,
            native_gradients,
            learning_rate,
            inplace=inplace,
        )

    def update_native_parameters(
        self,
        parameters: Sequence[LazyNode],
        gradients: Sequence[object],
        learning_rate: float,
        *,
        inplace: bool = False,
    ) -> Tuple[Tuple[LazyNode, ...], Tuple[LazyNode, ...]]:
        if len(parameters) != len(gradients):
            raise ValueError("parameter and gradient counts must match")
        native_parameters = tuple(self._load_input(node) for node in parameters)
        updated_values = []
        for start in range(0, len(parameters), self._SGD_FUSION_GROUP_SIZE):
            end = start + self._SGD_FUSION_GROUP_SIZE
            updated_values.extend(
                self._update_parameter_group(
                    native_parameters[start:end],
                    gradients[start:end],
                    learning_rate,
                    inplace=inplace,
                )
            )
        updated = tuple(updated_values)
        parameter_nodes = tuple(
            LazyNode.native_leaf("opencl", value, node.shape, node.dtype)
            for node, value in zip(parameters, updated)
        )
        gradient_nodes = tuple(
            LazyNode.native_leaf("opencl", value, node.shape, node.dtype)
            for node, value in zip(parameters, gradients)
        )
        return parameter_nodes, gradient_nodes

    def _update_parameter_group(
        self,
        parameters: Sequence[Any],
        gradients: Sequence[Any],
        learning_rate: float,
        *,
        inplace: bool,
    ) -> Tuple[Any, ...]:
        contiguous_parameters = tuple(self._contiguous(value) for value in parameters)
        contiguous_gradients = tuple(self._contiguous(value) for value in gradients)
        if any(
            value.dtype != np.float32
            for value in contiguous_parameters + contiguous_gradients
        ):
            raise TypeError("OpenCL SGD requires float32")
        outputs = (
            contiguous_parameters
            if inplace
            else tuple(
                self._empty(tuple(parameter.shape))
                for parameter in contiguous_parameters
            )
        )
        sizes = tuple(int(parameter.size) for parameter in contiguous_parameters)
        total = sum(sizes)
        if total == 0:
            return outputs

        declarations = []
        branches = []
        scalar_arg_dtypes = []
        arguments = []
        offset = 0
        for index, (parameter, gradient, output, size) in enumerate(
            zip(contiguous_parameters, contiguous_gradients, outputs, sizes)
        ):
            declarations.extend(
                (
                    f"    __global {'float' if inplace else 'const float'}* "
                    f"parameter_{index}",
                    f"    const ulong parameter_offset_{index}",
                    f"    __global const float* gradient_{index}",
                    f"    const ulong gradient_offset_{index}",
                )
            )
            scalar_arg_dtypes.extend((None, np.uint64, None, np.uint64))
            arguments.extend(
                (
                    parameter.data,
                    np.uint64(parameter.offset // parameter.dtype.itemsize),
                    gradient.data,
                    np.uint64(gradient.offset // gradient.dtype.itemsize),
                )
            )
            output_name = f"parameter_{index}"
            output_offset_name = f"parameter_offset_{index}"
            if not inplace:
                declarations.extend(
                    (
                        f"    __global float* output_{index}",
                        f"    const ulong output_offset_{index}",
                    )
                )
                scalar_arg_dtypes.extend((None, np.uint64))
                arguments.extend(
                    (
                        output.data,
                        np.uint64(output.offset // output.dtype.itemsize),
                    )
                )
                output_name = f"output_{index}"
                output_offset_name = f"output_offset_{index}"
            end = offset + size
            branches.append(
                f"    if (gid < {end}) {{\n"
                f"        const ulong index = gid - {offset};\n"
                f"        {output_name}[{output_offset_name} + index] = "
                f"parameter_{index}[parameter_offset_{index} + index] - "
                f"learning_rate * "
                f"gradient_{index}[gradient_offset_{index} + index];\n"
                f"        return;\n"
                f"    }}"
            )
            offset = end
        declarations.append("    const float learning_rate")
        scalar_arg_dtypes.append(np.float32)
        arguments.append(np.float32(learning_rate))
        source = f"""
__kernel void sgd_update_group(
{',\n'.join(declarations)}
) {{
    const ulong gid = get_global_id(0);
{chr(10).join(branches)}
}}
"""
        kernel = self._build_kernel(
            ("sgd_update_group", sizes, inplace),
            source,
            "sgd_update_group",
            scalar_arg_dtypes,
        )
        event = kernel(
            self.queue,
            (total,),
            None,
            *arguments,
        )
        self._record_event(
            event, *contiguous_parameters, *contiguous_gradients, *outputs
        )
        return outputs
