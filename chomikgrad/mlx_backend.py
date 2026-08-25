from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from .lazy import Compiler, CompiledProgram, LazyNode, Op, topological_sort


@dataclass
class MLXProgram(CompiledProgram):
    source: str
    inputs: Tuple[LazyNode, ...]
    _run: Callable[[Sequence[Any]], Tuple[Any, ...]]
    _mx: Any
    _load_input: Callable[[LazyNode], Any]

    def run_native(self) -> Tuple[Any, ...]:
        gpu = self._mx.Device(self._mx.gpu, 0)
        with self._mx.stream(gpu):
            values = [self._load_input(node) for node in self.inputs]
            outputs = self._run(values)
            self._mx.eval(outputs)
        return tuple(outputs)

    def __call__(self) -> Tuple[np.ndarray, ...]:
        outputs = self.run_native()
        return tuple(np.array(output) for output in outputs)


class MLXCompiler(Compiler):
    """Translate the five-operation IR to MLX and execute it on Metal GPU."""

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
        self._program_cache: Dict[object, Tuple[str, Callable[..., Any]]] = {}

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
        native = self._mx.array(value)
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

    def compile(self, outputs: Sequence[LazyNode]) -> MLXProgram:
        if not outputs:
            raise ValueError("at least one output is required")

        nodes = topological_sort(outputs)
        names = {node: f"v{index}" for index, node in enumerate(nodes)}
        leaves = tuple(node for node in nodes if node.op is None)
        leaf_indexes = {node: index for index, node in enumerate(leaves)}
        signature = self._signature(nodes, outputs)
        cached = self._program_cache.get(signature)
        if cached is not None:
            source, run = cached
            return MLXProgram(source, leaves, run, self._mx, self._load_input)

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
            compile(source, "<chomikgrad-mlx>", "exec"),
            {"mx": self._mx},
            namespace,
        )
        run = self._mx.compile(namespace["run"])
        self._program_cache[signature] = source, run
        return MLXProgram(  # type: ignore[arg-type]
            source,
            leaves,
            run,
            self._mx,
            self._load_input,
        )

    def update_parameters(
        self,
        parameters: Sequence[LazyNode],
        gradients: Sequence[LazyNode],
        learning_rate: float,
    ) -> Tuple[Tuple[LazyNode, ...], Tuple[LazyNode, ...]]:
        native_gradients = self.compile(gradients).run_native()
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
            return f"mx.{kind}({args[0]}, axis={axes!r}, keepdims={keepdims!r})"

        if node.op is Op.RESHAPE:
            return f"mx.reshape({args[0]}, {node.arg!r})"

        if node.op is Op.PERMUTE:
            return f"mx.transpose({args[0]}, axes={node.arg!r})"

        if node.op is Op.MATMUL:
            return f"mx.matmul({args[0]}, {args[1]})"

        raise ValueError(f"unsupported operation: {node.op}")
