from __future__ import annotations

from dataclasses import dataclass
import json
import mmap
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .lazy import (
    Compiler,
    CompiledProgram,
    DeviceAdapter,
    LazyNode,
    Op,
    topological_sort,
)


def _load_fp16_safetensors(path: Path) -> Dict[str, np.ndarray]:
    """Load floating-point safetensors directly as FP16 NumPy arrays."""
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_size))
        data_start = 8 + header_size
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            tensors: Dict[str, np.ndarray] = {}
            for name, metadata in header.items():
                if name == "__metadata__":
                    continue
                start, end = metadata["data_offsets"]
                raw = mapped[data_start + start : data_start + end]
                source_dtype = metadata["dtype"]
                shape = tuple(int(size) for size in metadata["shape"])
                if source_dtype == "BF16":
                    bits = np.frombuffer(raw, dtype="<u2")
                    fp32 = (bits.astype(np.uint32) << 16).view(np.float32)
                    value = fp32.astype(np.float16).reshape(shape)
                elif source_dtype == "F16":
                    value = np.frombuffer(raw, dtype="<f2").reshape(shape).copy()
                elif source_dtype == "F32":
                    value = (
                        np.frombuffer(raw, dtype="<f4")
                        .astype(np.float16)
                        .reshape(shape)
                    )
                else:
                    raise TypeError(
                        f"unsupported Core ML weight dtype {source_dtype!r} "
                        f"for {name!r}"
                    )
                tensors[name] = value
    return tensors


class CoreMLDeviceAdapter(DeviceAdapter):
    """Host-side values for synchronous Core ML prediction."""

    name = "coreml"
    inference_dtype = np.dtype(np.float16)

    def array(self, value: object) -> np.ndarray:
        return np.asarray(value)

    def evaluate(self, values: Sequence[object]) -> None:
        # MLModel.predict is synchronous.
        pass

    def synchronize(self) -> None:
        # MLModel.predict is synchronous.
        pass

    def to_numpy(self, value: object) -> np.ndarray:
        return np.asarray(value)

    def argmax(self, value: object) -> int:
        return int(np.asarray(value).argmax())

    def dtype(self, value: object) -> np.dtype:
        return np.asarray(value).dtype

    def load_safetensors(
        self, path: Path, *, dtype: Optional[np.dtype] = None
    ) -> Mapping[str, object]:
        if dtype is not None and np.dtype(dtype) != np.dtype(np.float16):
            raise TypeError("the Apple Neural Engine backend requires FP16 weights")
        return _load_fp16_safetensors(path)


@dataclass
class _CoreMLSegment:
    inputs: Tuple[LazyNode, ...]
    _input_names: Tuple[str, ...]
    outputs: Tuple[LazyNode, ...]
    _output_names: Tuple[str, ...]
    _model: Any


@dataclass
class CoreMLProgram(CompiledProgram):
    source: str
    inputs: Tuple[LazyNode, ...]
    outputs: Tuple[LazyNode, ...]
    _segments: Tuple[_CoreMLSegment, ...]
    device: CoreMLDeviceAdapter

    def run(
        self,
        bindings: Optional[Mapping[LazyNode, object]] = None,
        *,
        synchronize: bool = False,
    ) -> Tuple[object, ...]:
        replacements = bindings or {}
        environment: Dict[LazyNode, np.ndarray] = {}
        for node in self.inputs:
            if node in replacements:
                value = replacements[node]
            elif self.device.name in node.native_values:
                value = node.native_values[self.device.name]
            else:
                value = node.numpy_value()
            environment[node] = np.asarray(value, dtype=node.dtype)

        for segment in self._segments:
            prediction = segment._model.predict(
                {
                    name: environment[node]
                    for name, node in zip(segment._input_names, segment.inputs)
                }
            )
            for name, node in zip(segment._output_names, segment.outputs):
                environment[node] = np.asarray(
                    prediction[name], dtype=node.dtype
                )
        return tuple(environment[node] for node in self.outputs)

    def compute_plan_summary(self) -> Dict[str, object]:
        """Report Core ML's preferred device assignment for compiled operations."""
        from coremltools.models.compute_plan import MLComputePlan

        preferred: Dict[str, int] = {}
        neural_engine_supported = 0

        def visit(plan: Any, block: Any) -> None:
            nonlocal neural_engine_supported
            for operation in block.operations:
                for child in operation.blocks:
                    visit(plan, child)
                if operation.operator_name == "const":
                    continue
                usage = plan.get_compute_device_usage_for_mlprogram_operation(
                    operation
                )
                if usage is None:
                    continue
                label = type(usage.preferred_compute_device).__name__
                preferred[label] = preferred.get(label, 0) + 1
                if any(
                    type(device).__name__ == "MLNeuralEngineComputeDevice"
                    for device in usage.supported_compute_devices
                ):
                    neural_engine_supported += 1

        for segment in self._segments:
            plan = MLComputePlan.load_from_path(
                segment._model.get_compiled_model_path(),
                compute_units=segment._model.compute_unit,
            )
            structure = plan.model_structure.program
            if structure is None:
                continue
            for function in structure.functions.values():
                visit(plan, function.block)
        return {
            "compute_units": "CPU_AND_NE",
            "segments": len(self._segments),
            "preferred": preferred,
            "neural_engine_supported_operations": neural_engine_supported,
        }


class CoreMLCompiler(Compiler):
    """Compile the six-operation inference IR to a FP16 Core ML ML Program."""

    # Larger monolithic programs silently lose ANE eligibility on M1. A 1 GiB
    # constant budget keeps TinyLlama in three device-specialized segments.
    _MAX_SEGMENT_CONSTANT_BYTES = 1024 * 1024 * 1024

    def __init__(self) -> None:
        try:
            import coremltools as ct
            from coremltools.converters.mil import Builder as mb
            from coremltools.converters.mil import Function, Program
            from coremltools.converters.mil.mil import types
        except ImportError as error:
            raise ImportError(
                "the Core ML backend requires Apple silicon, macOS 15+ and "
                "`python -m pip install 'chomik-grad[coreml]'`"
            ) from error
        self._ct = ct
        self._mb = mb
        self._Function = Function
        self._Program = Program
        self._types = types
        self.device = CoreMLDeviceAdapter()

    def compile(
        self,
        outputs: Sequence[LazyNode],
        dynamic_inputs: Sequence[LazyNode] = (),
    ) -> CoreMLProgram:
        if not outputs:
            raise ValueError("at least one output is required")
        if any(node.op is not None for node in dynamic_inputs):
            raise ValueError("dynamic inputs must be graph leaves")

        nodes = topological_sort(outputs)
        leaves = tuple(node for node in nodes if node.op is None)
        lowered_leaves = {
            node
            for node in topological_sort(outputs, use_lowerings=True)
            if node.op is None
        }
        available_leaves = set(leaves) | lowered_leaves
        requested_dynamic = set(dynamic_inputs)
        if not requested_dynamic.issubset(available_leaves):
            raise ValueError("dynamic input is not a leaf of the compiled graph")
        program_inputs = (
            tuple(node for node in leaves if node in requested_dynamic)
            if dynamic_inputs
            else tuple(node for node in leaves if node.shape)
        )
        if any(not node.shape for node in program_inputs):
            raise ValueError("Core ML does not support scalar model inputs")
        dynamic_set = set(program_inputs)
        self._validate_dtypes(nodes)

        names = {node: f"v{index}" for index, node in enumerate(nodes)}
        dynamic_nodes: Dict[LazyNode, bool] = {}
        for node in nodes:
            dynamic_nodes[node] = (
                node in dynamic_set
                if node.op is None
                else any(dynamic_nodes[parent] for parent in node.inputs)
            )
        operation_groups = self._partition(nodes, dynamic_nodes)
        consumers: Dict[LazyNode, list[LazyNode]] = {node: [] for node in nodes}
        for node in nodes:
            for parent in node.inputs:
                consumers[parent].append(node)
        indexes = {node: index for index, node in enumerate(nodes)}
        requested_outputs = set(outputs)
        segments = []
        source_lines = [f"segments = {len(operation_groups)}"]

        for segment_index, operations in enumerate(operation_groups):
            operation_set = set(operations)
            segment_inputs = {
                parent
                for node in operations
                for parent in node.inputs
                if dynamic_nodes[parent] and parent not in operation_set
            }
            if not operations:
                segment_inputs.update(
                    node for node in outputs if dynamic_nodes[node]
                )
            ordered_inputs = tuple(sorted(segment_inputs, key=indexes.__getitem__))
            if any(not node.shape for node in ordered_inputs):
                raise ValueError("Core ML does not support scalar segment inputs")
            segment_outputs = {
                node
                for node in operations
                if node in requested_outputs
                or any(consumer not in operation_set for consumer in consumers[node])
            }
            if not operations:
                segment_outputs.update(outputs)
            ordered_outputs = tuple(
                sorted(segment_outputs, key=indexes.__getitem__)
            )
            segment = self._compile_segment(
                segment_index,
                operations,
                ordered_inputs,
                ordered_outputs,
                names,
                dynamic_nodes,
                dynamic_set,
            )
            segments.append(segment)
            source_lines.append(
                f"segment_{segment_index}: inputs={len(ordered_inputs)}, "
                f"ops={len(operations)}, outputs={len(ordered_outputs)}"
            )

        return CoreMLProgram(
            "\n".join(source_lines),
            program_inputs,
            tuple(outputs),
            tuple(segments),
            self.device,
        )

    def update_parameters(
        self,
        parameters: Sequence[LazyNode],
        gradients: Sequence[LazyNode],
        learning_rate: float,
        *,
        inplace: bool = False,
    ) -> None:
        raise RuntimeError("the Core ML backend is inference-only")

    def _partition(
        self,
        nodes: Sequence[LazyNode],
        dynamic_nodes: Mapping[LazyNode, bool],
    ) -> Tuple[Tuple[LazyNode, ...], ...]:
        groups = []
        current = []
        constant_bytes = 0
        for node in nodes:
            if node.op is None:
                if dynamic_nodes[node]:
                    continue
                size = int(np.prod(node.shape, dtype=np.int64)) * node.dtype.itemsize
                if (
                    current
                    and constant_bytes + size
                    > self._MAX_SEGMENT_CONSTANT_BYTES
                ):
                    groups.append(tuple(current))
                    current = []
                    constant_bytes = 0
                constant_bytes += size
            else:
                current.append(node)
        if current:
            groups.append(tuple(current))
        return tuple(groups or [tuple()])

    def _compile_segment(
        self,
        segment_index: int,
        operations: Sequence[LazyNode],
        inputs: Tuple[LazyNode, ...],
        outputs: Tuple[LazyNode, ...],
        names: Mapping[LazyNode, str],
        dynamic_nodes: Mapping[LazyNode, bool],
        dynamic_leaves: set[LazyNode],
    ) -> _CoreMLSegment:
        input_names = tuple(
            f"s{segment_index}_input_{index}" for index in range(len(inputs))
        )
        placeholders = {
            name: self._mb.placeholder(
                shape=node.shape,
                dtype=self._mil_dtype(node.dtype),
                name=name,
            )
            for name, node in zip(input_names, inputs)
        }
        program = self._Program()
        operation_set = set(operations)

        with self._Function(
            placeholders, opset_version=self._ct.target.macOS15
        ) as function:
            values: Dict[LazyNode, Any] = {
                node: function.inputs[name]
                for name, node in zip(input_names, inputs)
            }

            def resolve(node: LazyNode) -> Any:
                if node in values:
                    return values[node]
                if dynamic_nodes[node] and node not in operation_set:
                    raise RuntimeError("missing dynamic Core ML segment input")
                if node.op is None:
                    values[node] = self._mb.const(
                        val=self._constant_value(node), name=names[node]
                    )
                else:
                    for parent in node.inputs:
                        resolve(parent)
                    values[node] = self._operation(
                        node, values, names, dynamic_leaves
                    )
                return values[node]

            for node in operations:
                resolve(node)
            output_names = tuple(
                f"s{segment_index}_output_{index}"
                for index in range(len(outputs))
            )
            function.set_outputs(
                [
                    self._mb.identity(x=resolve(node), name=name)
                    for name, node in zip(output_names, outputs)
                ]
            )
        program.add_function("main", function)
        model = self._ct.convert(
            program,
            convert_to="mlprogram",
            minimum_deployment_target=self._ct.target.macOS15,
            compute_precision=self._ct.precision.FLOAT16,
            compute_units=self._ct.ComputeUnit.CPU_AND_NE,
        )
        return _CoreMLSegment(inputs, input_names, outputs, output_names, model)

    def _validate_dtypes(self, nodes: Sequence[LazyNode]) -> None:
        supported = {np.dtype(np.float16), np.dtype(np.int32), np.dtype(np.bool_)}
        for node in nodes:
            if node.dtype not in supported:
                raise TypeError(
                    "the Apple Neural Engine backend supports FP16, int32 and "
                    f"bool tensors; got {node.dtype}"
                )

    def _mil_dtype(self, dtype: np.dtype) -> Any:
        mapping = {
            np.dtype(np.float16): self._types.fp16,
            np.dtype(np.int32): self._types.int32,
            np.dtype(np.bool_): self._types.bool,
        }
        return mapping[np.dtype(dtype)]

    @staticmethod
    def _mil_dtype_name(dtype: np.dtype) -> str:
        mapping = {
            np.dtype(np.float16): "fp16",
            np.dtype(np.int32): "int32",
            np.dtype(np.bool_): "bool",
        }
        return mapping[np.dtype(dtype)]

    def _constant_value(self, node: LazyNode) -> np.ndarray:
        if self.device.name in node.native_values:
            value = node.native_values[self.device.name]
        else:
            value = node.numpy_value()
        return np.asarray(value, dtype=node.dtype)

    def _operation(
        self,
        node: LazyNode,
        values: Mapping[LazyNode, Any],
        names: Mapping[LazyNode, str],
        dynamic: set[LazyNode],
    ) -> Any:
        assert node.op is not None
        name = names[node]
        inputs = node.inputs
        args = [values[parent] for parent in inputs]

        if node.op is Op.ELEMENTWISE:
            kind = str(node.arg)
            if kind == "add":
                return self._mb.add(x=args[0], y=args[1], name=name)
            if kind == "sub":
                return self._mb.sub(x=args[0], y=args[1], name=name)
            if kind == "mul":
                return self._mb.mul(x=args[0], y=args[1], name=name)
            if kind == "div":
                return self._mb.real_div(x=args[0], y=args[1], name=name)
            if kind == "neg":
                return self._mb.mul(x=args[0], y=np.float16(-1), name=name)
            if kind == "exp":
                return self._mb.exp(x=args[0], name=name)
            if kind == "log":
                return self._mb.log(x=args[0], name=name)
            if kind == "sqrt":
                return self._mb.sqrt(x=args[0], name=name)
            if kind == "relu":
                return self._mb.relu(x=args[0], name=name)
            if kind in ("equal", "step"):
                compared = (
                    self._mb.equal(x=args[0], y=args[1], name=f"{name}_bool")
                    if kind == "equal"
                    else self._mb.greater(
                        x=args[0], y=np.float16(0), name=f"{name}_bool"
                    )
                )
                return self._mb.cast(
                    x=compared,
                    dtype=self._mil_dtype_name(node.dtype),
                    name=name,
                )
            raise ValueError(f"unsupported elementwise operation: {kind}")

        if node.op is Op.REDUCE:
            kind, axes, keepdims = node.arg
            operation = self._mb.reduce_sum if kind == "sum" else self._mb.reduce_max
            if kind not in ("sum", "max"):
                raise ValueError(f"unsupported reduction: {kind}")
            return operation(
                x=args[0], axes=list(axes), keep_dims=keepdims, name=name
            )

        if node.op is Op.RESHAPE:
            return self._mb.reshape(x=args[0], shape=list(node.arg), name=name)

        if node.op is Op.PERMUTE:
            return self._mb.transpose(x=args[0], perm=list(node.arg), name=name)

        if node.op is Op.MATMUL:
            weight_transpose = inputs[1]
            if (
                weight_transpose.op is Op.PERMUTE
                and weight_transpose.arg == (1, 0)
                and weight_transpose.inputs[0].op is None
                and weight_transpose.inputs[0] not in dynamic
            ):
                weight = values[weight_transpose.inputs[0]]
                bias = np.zeros(node.shape[-1], dtype=np.float16)
                linear_input = args[0]
                if len(inputs[0].shape) > 2:
                    linear_input = self._mb.reshape(
                        x=linear_input,
                        shape=[-1, inputs[0].shape[-1]],
                        name=f"{name}_flatten",
                    )
                linear = self._mb.linear(
                    x=linear_input,
                    weight=weight,
                    bias=bias,
                    name=name if len(inputs[0].shape) == 2 else f"{name}_linear",
                )
                if len(inputs[0].shape) > 2:
                    return self._mb.reshape(
                        x=linear, shape=list(node.shape), name=name
                    )
                return linear
            return self._mb.matmul(x=args[0], y=args[1], name=name)

        if node.op is Op.GATHER:
            return self._mb.gather(x=args[0], indices=args[1], axis=0, name=name)

        raise ValueError(f"unsupported operation: {node.op}")
