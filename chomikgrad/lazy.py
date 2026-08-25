from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np


class Op(Enum):
    """The complete low-level instruction set.

    Variants such as add/exp/relu are arguments of ELEMENTWISE rather than
    separate graph operations. This keeps compiler plugins deliberately small.
    """

    ELEMENTWISE = "elementwise"
    REDUCE = "reduce"
    RESHAPE = "reshape"
    PERMUTE = "permute"
    MATMUL = "matmul"
    GATHER = "gather"


class LazyNode:
    __slots__ = (
        "op",
        "inputs",
        "arg",
        "shape",
        "dtype",
        "value",
        "native_values",
        "cache_native",
        "lowering",
    )

    def __init__(
        self,
        op: Optional[Op],
        inputs: Sequence["LazyNode"],
        arg: object,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        value: Optional[np.ndarray] = None,
        native_values: Optional[Dict[str, object]] = None,
        cache_native: bool = True,
        lowering: Optional[Tuple[str, Tuple["LazyNode", ...], object]] = None,
    ) -> None:
        self.op = op
        self.inputs = tuple(inputs)
        self.arg = arg
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.value = value
        self.native_values = native_values or {}
        self.cache_native = cache_native
        self.lowering = lowering

    @classmethod
    def leaf(cls, value: np.ndarray, *, cache_native: bool = True) -> "LazyNode":
        return cls(
            None,
            (),
            None,
            value.shape,
            value.dtype,
            value,
            cache_native=cache_native,
        )

    @classmethod
    def native_leaf(
        cls,
        backend: str,
        value: object,
        shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> "LazyNode":
        return cls(None, (), None, shape, dtype, native_values={backend: value})

    def numpy_value(self) -> np.ndarray:
        if self.value is None:
            if not self.native_values:
                raise RuntimeError("leaf node has no value")
            self.value = np.array(next(iter(self.native_values.values())), copy=True)
        return self.value


class DeviceAdapter(ABC):
    """Small runtime boundary shared by compiled device backends."""

    name: str

    @abstractmethod
    def array(self, value: object) -> object:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, values: Sequence[object]) -> None:
        raise NotImplementedError

    @abstractmethod
    def synchronize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def to_numpy(self, value: object) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def argmax(self, value: object) -> int:
        raise NotImplementedError

    def argmax_last_axis(self, value: object) -> np.ndarray:
        """Return device results while keeping argmax outside the tensor IR."""
        return np.asarray(self.to_numpy(value)).argmax(axis=-1)

    @abstractmethod
    def dtype(self, value: object) -> np.dtype:
        raise NotImplementedError

    def load_safetensors(
        self, path: Path, *, dtype: Optional[np.dtype] = None
    ) -> Mapping[str, object]:
        raise NotImplementedError(
            f"the {self.name!r} device does not support safetensors loading"
        )

    def reset_peak_memory(self) -> None:
        pass

    def peak_memory_bytes(self) -> Optional[int]:
        return None


class CompiledProgram(ABC):
    source: str
    device: DeviceAdapter

    @abstractmethod
    def run(
        self,
        bindings: Optional[Mapping[LazyNode, object]] = None,
        *,
        synchronize: bool = False,
    ) -> Tuple[object, ...]:
        raise NotImplementedError

    def __call__(self) -> Tuple[np.ndarray, ...]:
        outputs = self.run(synchronize=True)
        return tuple(self.device.to_numpy(output) for output in outputs)


class Compiler(ABC):
    """A plugin translating lazy nodes into an executable program."""

    device: DeviceAdapter

    @abstractmethod
    def compile(
        self,
        outputs: Sequence[LazyNode],
        dynamic_inputs: Sequence[LazyNode] = (),
    ) -> CompiledProgram:
        raise NotImplementedError

    def update_parameters(
        self,
        parameters: Sequence[LazyNode],
        gradients: Sequence[LazyNode],
        learning_rate: float,
    ) -> Optional[Tuple[Tuple[LazyNode, ...], Tuple[LazyNode, ...]]]:
        return None


def topological_sort(
    outputs: Iterable[LazyNode], *, use_lowerings: bool = False
) -> Tuple[LazyNode, ...]:
    ordered = []
    visited = set()

    def visit(node: LazyNode) -> None:
        if node in visited:
            return
        visited.add(node)
        parents = (
            node.lowering[1]
            if use_lowerings and node.lowering is not None
            else node.inputs
        )
        for parent in parents:
            visit(parent)
        ordered.append(node)

    for output in outputs:
        visit(output)
    return tuple(ordered)


@dataclass
class NumpyProgram(CompiledProgram):
    source: str
    inputs: Tuple[LazyNode, ...]
    _run: Callable[[Sequence[np.ndarray]], Tuple[np.ndarray, ...]]
    device: DeviceAdapter

    def run(
        self,
        bindings: Optional[Mapping[LazyNode, object]] = None,
        *,
        synchronize: bool = False,
    ) -> Tuple[object, ...]:
        replacements = bindings or {}
        values = [
            np.asarray(replacements[node])
            if node in replacements
            else node.numpy_value()
            for node in self.inputs
        ]
        outputs = tuple(np.asarray(value) for value in self._run(values))
        if synchronize:
            self.device.evaluate(outputs)
        return outputs


class NumpyDeviceAdapter(DeviceAdapter):
    name = "cpu"

    def array(self, value: object) -> np.ndarray:
        return np.asarray(value)

    def evaluate(self, values: Sequence[object]) -> None:
        for value in values:
            np.asarray(value)

    def synchronize(self) -> None:
        pass

    def to_numpy(self, value: object) -> np.ndarray:
        return np.asarray(value)

    def argmax(self, value: object) -> int:
        return int(np.asarray(value).argmax())

    def dtype(self, value: object) -> np.dtype:
        return np.asarray(value).dtype


class NumpyCompiler(Compiler):
    """CPU compiler generating a straight-line Python/NumPy function."""

    _BINARY = {
        "add": "np.add({0}, {1})",
        "sub": "np.subtract({0}, {1})",
        "mul": "np.multiply({0}, {1})",
        "div": "np.divide({0}, {1})",
        "equal": "np.equal({0}, {1}).astype({0}.dtype, copy=False)",
    }
    _UNARY = {
        "neg": "np.negative({0})",
        "exp": "np.exp({0})",
        "log": "np.log({0})",
        "sqrt": "np.sqrt({0})",
        "relu": "np.maximum({0}, 0)",
        "step": "np.greater({0}, 0).astype({0}.dtype, copy=False)",
    }

    def __init__(self) -> None:
        self.device = NumpyDeviceAdapter()

    def compile(
        self,
        outputs: Sequence[LazyNode],
        dynamic_inputs: Sequence[LazyNode] = (),
    ) -> NumpyProgram:
        if not outputs:
            raise ValueError("at least one output is required")

        nodes = topological_sort(outputs)
        names = {node: f"v{index}" for index, node in enumerate(nodes)}
        leaves = tuple(node for node in nodes if node.op is None)
        leaf_indexes = {node: index for index, node in enumerate(leaves)}
        lines = ["def run(inputs):"]

        for node in nodes:
            name = names[node]
            if node.op is None:
                lines.append(f"    {name} = inputs[{leaf_indexes[node]}]")
                continue

            args = [names[parent] for parent in node.inputs]
            expression = self._expression(node, args)
            lines.append(f"    {name} = {expression}")

        rendered_outputs = ", ".join(names[node] for node in outputs)
        if len(outputs) == 1:
            rendered_outputs += ","
        lines.append(f"    return ({rendered_outputs})")
        source = "\n".join(lines)

        namespace: Dict[str, object] = {}
        exec(compile(source, "<chomikgrad-numpy>", "exec"), {"np": np}, namespace)
        return NumpyProgram(  # type: ignore[arg-type]
            source, leaves, namespace["run"], self.device
        )

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
            return f"np.{kind}({args[0]}, axis={axes!r}, keepdims={keepdims!r})"

        if node.op is Op.RESHAPE:
            return f"np.reshape({args[0]}, {node.arg!r})"

        if node.op is Op.PERMUTE:
            return f"np.transpose({args[0]}, axes={node.arg!r})"

        if node.op is Op.MATMUL:
            return f"np.matmul({args[0]}, {args[1]})"

        if node.op is Op.GATHER:
            return f"np.take({args[0]}, {args[1]}, axis=0)"

        raise ValueError(f"unsupported operation: {node.op}")


CompilerFactory = Callable[[], Compiler]
CompilerRegistration = Union[Compiler, CompilerFactory]
_COMPILERS: Dict[str, CompilerRegistration] = {}
_COMPILER_INSTANCES: Dict[str, Compiler] = {}
_DEFAULT_COMPILER = "cpu"


def register_compiler(
    name: str, compiler: CompilerRegistration, *, replace: bool = False
) -> None:
    if not name:
        raise ValueError("compiler name cannot be empty")
    if name in _COMPILERS and not replace:
        raise ValueError(f"compiler already registered: {name}")
    if not isinstance(compiler, Compiler) and not callable(compiler):
        raise TypeError("compiler must be a Compiler instance or a factory")
    _COMPILERS[name] = compiler
    _COMPILER_INSTANCES.pop(name, None)


def set_default_compiler(name: str) -> None:
    global _DEFAULT_COMPILER
    if name not in _COMPILERS:
        raise KeyError(f"unknown compiler: {name}")
    _DEFAULT_COMPILER = name


def get_compiler(name: Optional[str] = None) -> Compiler:
    selected = name or _DEFAULT_COMPILER
    if selected in _COMPILER_INSTANCES:
        return _COMPILER_INSTANCES[selected]
    try:
        registration = _COMPILERS[selected]
    except KeyError as error:
        raise KeyError(f"unknown compiler: {selected}") from error
    compiler = registration() if callable(registration) else registration
    if not isinstance(compiler, Compiler):
        raise TypeError(f"compiler factory {selected!r} returned an invalid object")
    _COMPILER_INSTANCES[selected] = compiler
    return compiler


def realize_nodes(
    outputs: Sequence[LazyNode], compiler: Optional[str] = None
) -> Tuple[np.ndarray, ...]:
    return get_compiler(compiler).compile(outputs)()


register_compiler("cpu", NumpyCompiler)
