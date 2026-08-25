from __future__ import annotations

import atexit
from dataclasses import dataclass
from pathlib import Path
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


class MLXDeviceAdapter(DeviceAdapter):
    name = "mlx"

    def __init__(self, mx: Any) -> None:
        self._mx = mx

    def array(self, value: object) -> Any:
        return self._mx.array(value)

    def evaluate(self, values: Sequence[object]) -> None:
        self._mx.eval(values)

    def synchronize(self) -> None:
        self._mx.synchronize()

    def to_numpy(self, value: object) -> np.ndarray:
        native = value
        if native.dtype == self._mx.bfloat16:  # type: ignore[attr-defined]
            native = native.astype(self._mx.float32)  # type: ignore[attr-defined]
        return np.array(native)

    def argmax(self, value: object) -> int:
        selected = self._mx.argmax(value)
        self._mx.eval(selected)
        return int(selected.item())

    def argmax_last_axis(self, value: object) -> np.ndarray:
        selected = self._mx.argmax(value, axis=-1)
        self._mx.eval(selected)
        return np.array(selected)

    def dtype(self, value: object) -> np.dtype:
        native_types = {
            self._mx.bool_: np.dtype(np.bool_),
            self._mx.int8: np.dtype(np.int8),
            self._mx.int16: np.dtype(np.int16),
            self._mx.int32: np.dtype(np.int32),
            self._mx.int64: np.dtype(np.int64),
            self._mx.uint8: np.dtype(np.uint8),
            self._mx.uint16: np.dtype(np.uint16),
            self._mx.uint32: np.dtype(np.uint32),
            self._mx.uint64: np.dtype(np.uint64),
            self._mx.float16: np.dtype(np.float16),
            self._mx.float32: np.dtype(np.float32),
            self._mx.float64: np.dtype(np.float64),
            self._mx.complex64: np.dtype(np.complex64),
        }
        native_dtype = getattr(value, "dtype", None)
        if native_dtype == self._mx.bfloat16:
            try:
                import ml_dtypes
            except ImportError as error:
                raise ImportError(
                    "BF16 metadata requires `python -m pip install ml-dtypes`"
                ) from error
            return np.dtype(ml_dtypes.bfloat16)
        try:
            return native_types[native_dtype]
        except KeyError as error:
            raise TypeError(f"unsupported MLX dtype: {value!r}") from error

    def load_safetensors(
        self, path: Path, *, dtype: Optional[np.dtype] = None
    ) -> Mapping[str, object]:
        native = self._mx.load(str(path))
        if not isinstance(native, dict):
            raise ValueError("expected a named safetensors weight file")
        if dtype is not None:
            requested = np.dtype(dtype)
            if requested == np.dtype(np.float16):
                target = self._mx.float16
            elif requested.name == "bfloat16":
                target = self._mx.bfloat16
            else:
                raise TypeError(f"unsupported MLX weight dtype: {requested}")
            native = {name: value.astype(target) for name, value in native.items()}
        return native

    def reset_peak_memory(self) -> None:
        self._mx.reset_peak_memory()

    def peak_memory_bytes(self) -> Optional[int]:
        return int(self._mx.get_peak_memory())


@dataclass
class MLXProgram(CompiledProgram):
    source: str
    inputs: Tuple[LazyNode, ...]
    _run: Callable[[Sequence[Any]], Tuple[Any, ...]]
    device: MLXDeviceAdapter
    _load_input: Callable[[LazyNode], Any]

    def run(
        self,
        bindings: Optional[Mapping[LazyNode, object]] = None,
        *,
        synchronize: bool = False,
    ) -> Tuple[object, ...]:
        mx = self.device._mx
        gpu = mx.Device(mx.gpu, 0)
        with mx.stream(gpu):
            replacements = bindings or {}
            values = [
                replacements[node]
                if node in replacements
                else self._load_input(node)
                for node in self.inputs
            ]
            outputs = self._run(values)
            if synchronize:
                self.device.evaluate(outputs)
        return tuple(outputs)

    def run_native(
        self,
        *,
        evaluate: bool = True,
        input_values: Optional[Mapping[LazyNode, Any]] = None,
    ) -> Tuple[Any, ...]:
        """Compatibility alias for the backend-neutral run method."""
        return self.run(  # type: ignore[return-value]
            input_values,
            synchronize=evaluate,
        )


class MLXCompiler(Compiler):
    """Translate the six-operation IR to MLX and execute it on Metal GPU."""

    _BINARY = {
        "add": "mx.add({0}, {1})",
        "sub": "mx.subtract({0}, {1})",
        "mul": "mx.multiply({0}, {1})",
        "div": "mx.divide({0}, {1})",
        "equal": "mx.equal({0}, {1}).astype({0}.dtype)",
    }
    _UNARY = {
        "neg": "mx.negative({0})",
        "exp": "mx.exp({0})",
        "log": "mx.log({0})",
        "sqrt": "mx.sqrt({0})",
        "relu": "mx.maximum({0}, 0)",
        "step": "mx.greater({0}, 0).astype({0}.dtype)",
    }

    def __init__(self) -> None:
        try:
            import mlx.core as mx
        except ImportError as error:
            raise ImportError(
                "the MLX backend requires Apple silicon, native Python 3.10+ "
                "and `python -m pip install 'chomik-grad[mlx]'`"
            ) from error
        if not mx.metal.is_available():
            raise RuntimeError("the MLX backend requires an available Metal GPU")
        self._mx = mx
        self.device = MLXDeviceAdapter(mx)
        self._program_cache: Dict[object, Tuple[str, Callable[..., Any]]] = {}
        atexit.register(self.close)

    def close(self) -> None:
        if not self._program_cache:
            return
        self.device.synchronize()
        self._program_cache.clear()
        self._mx.clear_cache()

    @property
    def cache_size(self) -> int:
        return len(self._program_cache)

    def _load_input(self, node: LazyNode) -> Any:
        if "mlx" in node.native_values:
            return node.native_values["mlx"]
        value = node.numpy_value()
        if value.dtype == np.float64:
            raise TypeError(
                "MLX does not support float64; create this tensor as float32"
            )
        native = self.device.array(value)
        if node.cache_native:
            node.native_values["mlx"] = native
        return native

    def _signature(
        self,
        nodes: Sequence[LazyNode],
        outputs: Sequence[LazyNode],
    ) -> object:
        indexes = {node: index for index, node in enumerate(nodes)}
        specifications = []
        for node in nodes:
            if node.op is None:
                specifications.append((None, node.shape, node.dtype.str))
            elif node.lowering is not None:
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

    def compile(
        self,
        outputs: Sequence[LazyNode],
        dynamic_inputs: Sequence[LazyNode] = (),
    ) -> MLXProgram:
        if not outputs:
            raise ValueError("at least one output is required")
        if any(node.op is not None for node in dynamic_inputs):
            raise ValueError("dynamic inputs must be graph leaves")
        requested_dynamic = set(dynamic_inputs)
        portable_leaves = {
            node for node in topological_sort(outputs) if node.op is None
        }
        lowered_leaves = {
            node
            for node in topological_sort(outputs, use_lowerings=True)
            if node.op is None
        }
        if not requested_dynamic.issubset(portable_leaves | lowered_leaves):
            raise ValueError("dynamic input is not a leaf of the compiled graph")

        nodes = topological_sort(outputs, use_lowerings=True)
        leaves = tuple(node for node in nodes if node.op is None)
        program_inputs = tuple(
            node for node in leaves if node in requested_dynamic
        )
        specialized = bool(dynamic_inputs)
        signature = self._signature(nodes, outputs)
        cached = None if specialized else self._program_cache.get(signature)
        if cached is not None:
            source, run = cached
            return MLXProgram(source, leaves, run, self.device, self._load_input)

        names = {node: f"v{index}" for index, node in enumerate(nodes)}
        leaf_indexes = {node: index for index, node in enumerate(leaves)}
        lines = ["def run(inputs):"]
        lowering_bundles: Dict[object, str] = {}

        for node in nodes:
            name = names[node]
            if node.op is None:
                lines.append(f"    {name} = inputs[{leaf_indexes[node]}]")
                continue
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
                    lowered = [names[parent] for parent in lowering_inputs]
                    lines.append(
                        f"    {bundle} = layer_norm_backward("
                        f"{lowered[0]}, {lowered[1]}, {lowered[2]}, "
                        f"{float(epsilon)!r})"
                    )
                lines.append(f"    {name} = {bundle}[{int(component)}]")
                continue
            args = (
                []
                if node.lowering is not None
                else [names[parent] for parent in node.inputs]
            )
            lines.append(
                f"    {name} = {self._expression(node, args, names)}"
            )

        rendered_outputs = ", ".join(names[node] for node in outputs)
        if len(outputs) == 1:
            rendered_outputs += ","
        lines.append(f"    return ({rendered_outputs})")
        source = "\n".join(lines)

        namespace: Dict[str, object] = {}
        exec(
            compile(source, "<chomikgrad-mlx>", "exec"),
            {
                "mx": self._mx,
                "layer_norm": self._layer_norm,
                "layer_norm_backward": self._layer_norm_backward,
                "softmax_backward": self._softmax_backward,
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

            def run_dynamic(values: Sequence[Any]) -> Tuple[Any, ...]:
                merged = list(template)
                for leaf_index, node in enumerate(leaves):
                    if node in dynamic_indexes:
                        merged[leaf_index] = values[dynamic_indexes[node]]
                return raw_run(merged)

            run = self._mx.compile(run_dynamic)
        else:
            run = self._mx.compile(raw_run)
            self._program_cache[signature] = source, run
        return MLXProgram(  # type: ignore[arg-type]
            source,
            program_inputs if specialized else leaves,
            run,
            self.device,
            self._load_input,
        )

    def update_parameters(
        self,
        parameters: Sequence[LazyNode],
        gradients: Sequence[LazyNode],
        learning_rate: float,
        *,
        inplace: bool = False,
    ) -> Tuple[Tuple[LazyNode, ...], Tuple[LazyNode, ...]]:
        if inplace:
            raise RuntimeError(
                "the MLX backend does not support in-place parameter updates"
            )
        native_gradients = self.compile(gradients).run(synchronize=False)
        native_parameters = [self._load_input(node) for node in parameters]
        gpu = self._mx.Device(self._mx.gpu, 0)
        with self._mx.stream(gpu):
            updated = tuple(
                parameter - learning_rate * gradient
                for parameter, gradient in zip(native_parameters, native_gradients)
            )
            self._mx.eval(updated, native_gradients)
        parameter_nodes = tuple(
            LazyNode.native_leaf("mlx", value, node.shape, node.dtype)
            for node, value in zip(parameters, updated)
        )
        gradient_nodes = tuple(
            LazyNode.native_leaf("mlx", value, node.shape, node.dtype)
            for node, value in zip(parameters, native_gradients)
        )
        return parameter_nodes, gradient_nodes

    def _expression(
        self,
        node: LazyNode,
        args: Sequence[str],
        names: Mapping[LazyNode, str],
    ) -> str:
        if node.lowering is not None:
            kind, inputs, argument = node.lowering
            lowered = [names[parent] for parent in inputs]
            if kind == "layer_norm":
                return (
                    f"layer_norm({lowered[0]}, {lowered[1]}, {lowered[2]}, "
                    f"{float(argument)!r})"
                )
            if kind == "softmax_backward":
                return (
                    f"softmax_backward({lowered[0]}, {lowered[1]}, "
                    f"{int(argument)!r})"
                )
            if kind == "layer_norm_backward":
                raise RuntimeError(
                    "layer_norm_backward must be emitted as a shared bundle"
                )
            if kind == "rms_norm":
                return (
                    f"mx.fast.rms_norm({lowered[0]}, {lowered[1]}, "
                    f"{float(argument)!r})"
                )
            if kind == "rope":
                dimensions, theta, offset = argument
                actual_offset = lowered[1] if len(lowered) == 2 else repr(offset)
                return (
                    f"mx.fast.rope({lowered[0]}, {int(dimensions)!r}, "
                    f"traditional=False, base={float(theta)!r}, scale=1.0, "
                    f"offset={actual_offset})"
                )
            if kind == "sdpa":
                return (
                    "mx.fast.scaled_dot_product_attention("
                    f"{lowered[0]}, {lowered[1]}, {lowered[2]}, "
                    f"scale={float(argument)!r}, mask={lowered[3]})"
                )
            raise ValueError(f"unsupported MLX lowering: {kind}")

        if node.op is Op.ELEMENTWISE:
            kind = str(node.arg)
            if kind in self._BINARY:
                return self._BINARY[kind].format(*args)
            if kind in self._UNARY:
                return self._UNARY[kind].format(*args)
            raise ValueError(f"unsupported elementwise operation: {kind}")

        if node.op is Op.REDUCE:
            kind, axes, keepdims = node.arg  # type: ignore[misc]
            if kind not in ("sum", "max"):
                raise ValueError(f"unsupported reduction: {kind}")
            return f"mx.{kind}({args[0]}, axis={axes!r}, keepdims={keepdims!r})"

        if node.op is Op.RESHAPE:
            return f"mx.reshape({args[0]}, {node.arg!r})"

        if node.op is Op.PERMUTE:
            return f"mx.transpose({args[0]}, axes={node.arg!r})"

        if node.op is Op.MATMUL:
            return f"mx.matmul({args[0]}, {args[1]})"

        if node.op is Op.GATHER:
            return f"mx.take({args[0]}, {args[1]}, axis=0)"

        raise ValueError(f"unsupported operation: {node.op}")

    def _layer_norm(
        self,
        inputs: Any,
        weight: Any,
        bias: Any,
        epsilon: float,
    ) -> Any:
        mean = self._mx.mean(inputs, axis=-1, keepdims=True)
        centered = inputs - mean
        variance = self._mx.mean(centered * centered, axis=-1, keepdims=True)
        return centered / self._mx.sqrt(variance + epsilon) * weight + bias

    def _softmax_backward(
        self,
        gradients: Any,
        outputs: Any,
        axis: int,
    ) -> Any:
        weighted = gradients * outputs
        projection = self._mx.sum(weighted, axis=axis, keepdims=True)
        return outputs * (gradients - projection)

    def _layer_norm_backward(
        self,
        gradients: Any,
        inputs: Any,
        weight: Any,
        epsilon: float,
    ) -> Tuple[Any, Any, Any]:
        mx = self._mx
        mean = mx.mean(inputs, axis=-1, keepdims=True)
        centered = inputs - mean
        variance = mx.mean(centered * centered, axis=-1, keepdims=True)
        inverse_std = 1.0 / mx.sqrt(variance + epsilon)
        normalized = centered * inverse_std
        weighted = gradients * weight
        input_gradients = (
            weighted
            - mx.mean(weighted, axis=-1, keepdims=True)
            - normalized
            * mx.mean(weighted * normalized, axis=-1, keepdims=True)
        ) * inverse_std
        rows = mx.reshape(gradients, (-1, gradients.shape[-1]))
        normalized_rows = mx.reshape(normalized, rows.shape)
        weight_gradients = mx.sum(rows * normalized_rows, axis=0)
        bias_gradients = mx.sum(rows, axis=0)
        return input_gradients, weight_gradients, bias_gradients
