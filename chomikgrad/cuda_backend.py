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


class CUDADeviceAdapter(DeviceAdapter):
    name = "cuda"

    def __init__(self, cp: Any) -> None:
        self._cp = cp

    def array(self, value: object) -> Any:
        return self._cp.asarray(value)

    def evaluate(self, values: Sequence[object]) -> None:
        self.synchronize()

    def synchronize(self) -> None:
        self._cp.cuda.get_current_stream().synchronize()

    def to_numpy(self, value: object) -> np.ndarray:
        return self._cp.asnumpy(value)

    def argmax(self, value: object) -> int:
        return int(self._cp.argmax(value).item())

    def argmax_last_axis(self, value: object) -> np.ndarray:
        return self._cp.asnumpy(self._cp.argmax(value, axis=-1))

    def dtype(self, value: object) -> np.dtype:
        return np.dtype(value.dtype)  # type: ignore[attr-defined]


@dataclass
class CUDAProgram(CompiledProgram):
    source: str
    inputs: Tuple[LazyNode, ...]
    _run: Callable[[Sequence[Any]], Tuple[Any, ...]]
    device: CUDADeviceAdapter
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


class CUDACompiler(Compiler):
    """Translate the portable six-operation IR to CuPy on an NVIDIA GPU."""

    _BINARY = {
        "add": "cp.add({0}, {1})",
        "sub": "cp.subtract({0}, {1})",
        "mul": "cp.multiply({0}, {1})",
        "div": "cp.divide({0}, {1})",
        "equal": "cp.equal({0}, {1}).astype({0}.dtype, copy=False)",
    }
    _UNARY = {
        "neg": "cp.negative({0})",
        "exp": "cp.exp({0})",
        "log": "cp.log({0})",
        "sqrt": "cp.sqrt({0})",
        "relu": "cp.maximum({0}, 0)",
        "step": "cp.greater({0}, 0).astype({0}.dtype, copy=False)",
    }

    def __init__(self) -> None:
        try:
            import cupy as cp
        except ImportError as error:
            raise ImportError(
                "the CUDA backend requires `python -m pip install "
                "'chomik-grad[cuda]'`"
            ) from error
        try:
            device_count = int(cp.cuda.runtime.getDeviceCount())
        except Exception as error:
            raise RuntimeError(
                "the CUDA backend requires a working NVIDIA driver and CUDA GPU"
            ) from error
        if device_count < 1:
            raise RuntimeError("the CUDA backend requires an available CUDA GPU")

        self._cp = cp
        self.device = CUDADeviceAdapter(cp)
        self._program_cache: Dict[object, Tuple[str, Callable[..., Any]]] = {}

    @property
    def cache_size(self) -> int:
        return len(self._program_cache)

    def close(self) -> None:
        if not self._program_cache:
            return
        self.device.synchronize()
        self._program_cache.clear()

    def _load_input(self, node: LazyNode) -> Any:
        if "cuda" in node.native_values:
            return node.native_values["cuda"]
        native = self.device.array(node.numpy_value())
        if node.cache_native:
            node.native_values["cuda"] = native
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
    ) -> CUDAProgram:
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
            return CUDAProgram(source, leaves, run, self.device, self._load_input)

        names = {node: f"v{index}" for index, node in enumerate(nodes)}
        leaf_indexes = {node: index for index, node in enumerate(leaves)}
        lines = ["def run(inputs):"]
        for node in nodes:
            name = names[node]
            if node.op is None:
                lines.append(f"    {name} = inputs[{leaf_indexes[node]}]")
                continue
            args = [names[parent] for parent in node.inputs]
            lines.append(f"    {name} = {self._expression(node, args)}")

        rendered_outputs = ", ".join(names[node] for node in outputs)
        if len(outputs) == 1:
            rendered_outputs += ","
        lines.append(f"    return ({rendered_outputs})")
        source = "\n".join(lines)

        namespace: Dict[str, object] = {}
        exec(
            compile(source, "<chomikgrad-cuda>", "exec"),
            {"cp": self._cp},
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

        return CUDAProgram(  # type: ignore[arg-type]
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
    ) -> Tuple[Tuple[LazyNode, ...], Tuple[LazyNode, ...]]:
        native_gradients = self.compile(gradients).run(synchronize=False)
        native_parameters = [self._load_input(node) for node in parameters]
        updated = tuple(
            parameter - learning_rate * gradient
            for parameter, gradient in zip(native_parameters, native_gradients)
        )
        parameter_nodes = tuple(
            LazyNode.native_leaf("cuda", value, node.shape, node.dtype)
            for node, value in zip(parameters, updated)
        )
        gradient_nodes = tuple(
            LazyNode.native_leaf("cuda", value, node.shape, node.dtype)
            for node, value in zip(parameters, native_gradients)
        )
        return parameter_nodes, gradient_nodes

    def _expression(self, node: LazyNode, args: Sequence[str]) -> str:
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
            return f"cp.{kind}({args[0]}, axis={axes!r}, keepdims={keepdims!r})"

        if node.op is Op.RESHAPE:
            return f"cp.reshape({args[0]}, {node.arg!r})"
        if node.op is Op.PERMUTE:
            return f"cp.transpose({args[0]}, axes={node.arg!r})"
        if node.op is Op.MATMUL:
            return f"cp.matmul({args[0]}, {args[1]})"
        if node.op is Op.GATHER:
            return f"cp.take({args[0]}, {args[1]}, axis=0)"
        raise ValueError(f"unsupported operation: {node.op}")
