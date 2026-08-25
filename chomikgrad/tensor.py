from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .lazy import CompiledProgram, LazyNode, Op, get_compiler


ArrayLike = Union[float, int, Sequence[float], np.ndarray]
Axis = Optional[Union[int, Sequence[int]]]
BackwardResult = Iterable[Tuple["Tensor", "Tensor"]]

_GRAD_ENABLED = True


@contextmanager
def no_grad() -> Iterator[None]:
    global _GRAD_ENABLED
    previous = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = previous


def _normalize_axes(axis: Axis, ndim: int) -> Tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    raw = (axis,) if isinstance(axis, int) else tuple(axis)
    normalized = tuple(item + ndim if item < 0 else item for item in raw)
    if any(item < 0 or item >= ndim for item in normalized):
        raise ValueError(f"axis {axis!r} is invalid for a {ndim}D tensor")
    if len(set(normalized)) != len(normalized):
        raise ValueError("reduction axes must be unique")
    return normalized


def _reduced_shape(
    shape: Tuple[int, ...], axes: Tuple[int, ...], keepdims: bool
) -> Tuple[int, ...]:
    if keepdims:
        return tuple(1 if index in axes else size for index, size in enumerate(shape))
    return tuple(size for index, size in enumerate(shape) if index not in axes)


class Tensor:
    def __init__(
        self,
        data: ArrayLike,
        *,
        requires_grad: bool = False,
        dtype: Optional[np.dtype] = None,
        copy: bool = True,
    ) -> None:
        value = np.asarray(data, dtype=dtype)
        if requires_grad and not np.issubdtype(value.dtype, np.floating):
            raise TypeError("only floating-point tensors can require gradients")
        self._node = LazyNode.leaf(
            value.copy() if copy else value,
            cache_native=copy,
        )
        self.requires_grad = bool(requires_grad)
        self.grad: Optional[Tensor] = None
        self._parents: Tuple[Tensor, ...] = ()
        self._backward: Callable[[Tensor], BackwardResult] = lambda grad: ()
        self._is_leaf = True

    @classmethod
    def _from_node(cls, node: LazyNode, requires_grad: bool) -> "Tensor":
        tensor = cls.__new__(cls)
        tensor._node = node
        tensor.requires_grad = requires_grad
        tensor.grad = None
        tensor._parents = ()
        tensor._backward = lambda grad: ()
        tensor._is_leaf = False
        return tensor

    @classmethod
    def from_native(
        cls,
        value: object,
        *,
        backend: str,
        shape: Sequence[int],
        dtype: np.dtype,
    ) -> "Tensor":
        """Wrap storage already owned by a compiler backend without copying it."""
        if not backend:
            raise ValueError("backend name cannot be empty")
        native_shape = tuple(int(size) for size in shape)
        if any(size < 0 for size in native_shape):
            raise ValueError("native tensor dimensions cannot be negative")
        tensor = cls._from_node(
            LazyNode.native_leaf(backend, value, native_shape, np.dtype(dtype)),
            requires_grad=False,
        )
        tensor._is_leaf = True
        return tensor

    @classmethod
    def zeros(
        cls, shape: Sequence[int], *, dtype: np.dtype = np.float32
    ) -> "Tensor":
        return cls(np.zeros(tuple(shape), dtype=dtype), copy=False)

    @classmethod
    def ones(
        cls, shape: Sequence[int], *, dtype: np.dtype = np.float32
    ) -> "Tensor":
        return cls(np.ones(tuple(shape), dtype=dtype), copy=False)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._node.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._node.dtype

    @property
    def T(self) -> "Tensor":
        if self.ndim != 2:
            raise ValueError("T is only defined for 2D tensors; use permute instead")
        return self.permute(1, 0)

    def numpy(self, compiler: Optional[str] = None) -> np.ndarray:
        return realize(self, compiler=compiler)[0]

    def item(self, compiler: Optional[str] = None) -> object:
        if self.shape != ():
            raise ValueError("item() requires a scalar tensor")
        return self.numpy(compiler).item()

    def detach(self) -> "Tensor":
        return Tensor._from_node(self._node, requires_grad=False)

    def assign(self, value: ArrayLike) -> None:
        if not self._is_leaf or self._node.value is None:
            raise RuntimeError("only leaf tensors can be assigned")
        replacement = np.asarray(value, dtype=self.dtype)
        if replacement.shape != self.shape:
            raise ValueError(f"expected shape {self.shape}, got {replacement.shape}")
        # Replacing storage preserves the meaning of lazy graphs created before
        # the assignment: they still reference the old leaf and its old value.
        self._node = LazyNode.leaf(replacement.copy())

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, grad: Optional[ArrayLike] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("cannot call backward() on a tensor without gradients")
        if grad is None:
            if self.shape != ():
                raise ValueError("a gradient is required for non-scalar outputs")
            initial = Tensor(np.asarray(1, dtype=self.dtype))
        else:
            initial = grad if isinstance(grad, Tensor) else Tensor(grad, dtype=self.dtype)
            if initial.shape != self.shape:
                raise ValueError(f"expected gradient shape {self.shape}, got {initial.shape}")

        ordered: List[Tensor] = []
        visited = set()

        def visit(tensor: Tensor) -> None:
            if tensor in visited:
                return
            visited.add(tensor)
            for parent in tensor._parents:
                visit(parent)
            ordered.append(tensor)

        visit(self)

        with no_grad():
            gradients = {self: initial}
            for tensor in reversed(ordered):
                incoming = gradients.get(tensor)
                if incoming is None:
                    continue
                for parent, contribution in tensor._backward(incoming):
                    if not parent.requires_grad:
                        continue
                    previous = gradients.get(parent)
                    gradients[parent] = (
                        contribution if previous is None else previous + contribution
                    )

            for tensor, contribution in gradients.items():
                if tensor.requires_grad and tensor._is_leaf:
                    tensor.grad = (
                        contribution
                        if tensor.grad is None
                        else tensor.grad + contribution
                    )

    def _coerce(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        if isinstance(other, Tensor):
            return other
        return Tensor(other, dtype=self.dtype)

    def _binary(self, other: Union["Tensor", ArrayLike], kind: str) -> "Tensor":
        right = self._coerce(other)
        shape = np.broadcast_shapes(self.shape, right.shape)
        dtype = np.result_type(self.dtype, right.dtype)
        node = LazyNode(Op.ELEMENTWISE, (self._node, right._node), kind, shape, dtype)
        tracks = _GRAD_ENABLED and kind != "equal" and (
            self.requires_grad or right.requires_grad
        )
        result = Tensor._from_node(node, tracks)
        if not tracks:
            return result

        result._parents = (self, right)

        def backward(grad: Tensor) -> BackwardResult:
            if kind == "add":
                left_grad, right_grad = grad, grad
            elif kind == "sub":
                left_grad, right_grad = grad, -grad
            elif kind == "mul":
                left_grad, right_grad = grad * right, grad * self
            elif kind == "div":
                left_grad = grad / right
                right_grad = -(grad * self) / (right * right)
            else:
                raise RuntimeError(f"no gradient for elementwise operation {kind}")
            contributions = []
            if self.requires_grad:
                contributions.append((self, left_grad._unbroadcast(self.shape)))
            if right.requires_grad:
                contributions.append((right, right_grad._unbroadcast(right.shape)))
            return contributions

        result._backward = backward
        return result

    def _unary(self, kind: str) -> "Tensor":
        node = LazyNode(Op.ELEMENTWISE, (self._node,), kind, self.shape, self.dtype)
        tracks = _GRAD_ENABLED and kind != "step" and self.requires_grad
        result = Tensor._from_node(node, tracks)
        if not tracks:
            return result

        result._parents = (self,)

        def backward(grad: Tensor) -> BackwardResult:
            if kind == "neg":
                contribution = -grad
            elif kind == "exp":
                contribution = grad * result
            elif kind == "log":
                contribution = grad / self
            elif kind == "sqrt":
                contribution = grad / (2 * result)
            elif kind == "relu":
                contribution = grad * self._unary("step")
            else:
                raise RuntimeError(f"no gradient for elementwise operation {kind}")
            return ((self, contribution),)

        result._backward = backward
        return result

    def _unbroadcast(self, shape: Tuple[int, ...]) -> "Tensor":
        if self.shape == shape:
            return self
        extra = len(self.shape) - len(shape)
        axes = list(range(extra))
        axes.extend(
            extra + index
            for index, size in enumerate(shape)
            if size == 1 and self.shape[extra + index] != 1
        )
        reduced = self.sum(axis=tuple(axes), keepdims=True) if axes else self
        return reduced.reshape(shape)

    def __add__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self._binary(other, "add")

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __sub__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self._binary(other, "sub")

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) - self

    def __mul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self._binary(other, "mul")

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __truediv__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self._binary(other, "div")

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) / self

    def __neg__(self) -> "Tensor":
        return self._unary("neg")

    def exp(self) -> "Tensor":
        return self._unary("exp")

    def log(self) -> "Tensor":
        return self._unary("log")

    def relu(self) -> "Tensor":
        return self._unary("relu")

    def sqrt(self) -> "Tensor":
        return self._unary("sqrt")

    def softmax(self, axis: int = -1) -> "Tensor":
        maximum = self.max(axis=axis, keepdims=True).detach()
        exponentials = (self - maximum).exp()
        return exponentials / exponentials.sum(axis=axis, keepdims=True)

    def log_softmax(self, axis: int = -1) -> "Tensor":
        maximum = self.max(axis=axis, keepdims=True).detach()
        shifted = self - maximum
        return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()

    def reshape(self, *shape: Union[int, Sequence[int]]) -> "Tensor":
        requested = tuple(shape[0]) if len(shape) == 1 and not isinstance(shape[0], int) else tuple(shape)  # type: ignore[arg-type]
        inferred = list(requested)
        if inferred.count(-1) > 1:
            raise ValueError("only one inferred dimension is allowed")
        if -1 in inferred:
            known = int(np.prod([size for size in inferred if size != -1]))
            total = int(np.prod(self.shape))
            if known == 0 or total % known:
                raise ValueError("shape is not compatible with tensor size")
            inferred[inferred.index(-1)] = total // known
        new_shape = tuple(int(size) for size in inferred)
        if int(np.prod(new_shape)) != int(np.prod(self.shape)):
            raise ValueError("shape is not compatible with tensor size")
        node = LazyNode(Op.RESHAPE, (self._node,), new_shape, new_shape, self.dtype)
        tracks = _GRAD_ENABLED and self.requires_grad
        result = Tensor._from_node(node, tracks)
        if tracks:
            result._parents = (self,)
            result._backward = lambda grad: ((self, grad.reshape(self.shape)),)
        return result

    def permute(self, *axes: int) -> "Tensor":
        if len(axes) != self.ndim or set(axes) != set(range(self.ndim)):
            raise ValueError(f"expected a permutation of {tuple(range(self.ndim))}")
        shape = tuple(self.shape[index] for index in axes)
        node = LazyNode(Op.PERMUTE, (self._node,), tuple(axes), shape, self.dtype)
        tracks = _GRAD_ENABLED and self.requires_grad
        result = Tensor._from_node(node, tracks)
        if tracks:
            inverse = tuple(np.argsort(axes).tolist())
            result._parents = (self,)
            result._backward = lambda grad: ((self, grad.permute(*inverse)),)
        return result

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        first = dim0 + self.ndim if dim0 < 0 else dim0
        second = dim1 + self.ndim if dim1 < 0 else dim1
        if first < 0 or first >= self.ndim or second < 0 or second >= self.ndim:
            raise ValueError(f"invalid transpose dimensions {dim0}, {dim1}")
        axes = list(range(self.ndim))
        axes[first], axes[second] = axes[second], axes[first]
        return self.permute(*axes)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TypeError("matmul requires another Tensor")
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError("matmul requires tensors with at least two dimensions")
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(f"cannot multiply shapes {self.shape} and {other.shape}")
        batch_shape = np.broadcast_shapes(self.shape[:-2], other.shape[:-2])
        shape = batch_shape + (self.shape[-2], other.shape[-1])
        dtype = np.result_type(self.dtype, other.dtype)
        node = LazyNode(Op.MATMUL, (self._node, other._node), None, shape, dtype)
        tracks = _GRAD_ENABLED and (self.requires_grad or other.requires_grad)
        result = Tensor._from_node(node, tracks)
        if tracks:
            result._parents = (self, other)

            def backward(grad: Tensor) -> BackwardResult:
                contributions = []
                if self.requires_grad:
                    left_grad = grad @ other.transpose(-2, -1)
                    contributions.append((self, left_grad._unbroadcast(self.shape)))
                if other.requires_grad:
                    right_grad = self.transpose(-2, -1) @ grad
                    contributions.append((other, right_grad._unbroadcast(other.shape)))
                return contributions

            result._backward = backward
        return result

    def gather(self, indices: "Tensor") -> "Tensor":
        """Select rows by integer index, with a dense portable gradient."""
        if not isinstance(indices, Tensor):
            raise TypeError("gather indices must be a Tensor")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("gather indices must have an integer dtype")
        shape = indices.shape + self.shape[1:]
        node = LazyNode(
            Op.GATHER,
            (self._node, indices._node),
            None,
            shape,
            self.dtype,
        )
        tracks = _GRAD_ENABLED and self.requires_grad
        result = Tensor._from_node(node, tracks)
        if not tracks:
            return result

        result._parents = (self,)

        def backward(grad: Tensor) -> BackwardResult:
            rows = self.shape[0]
            count = int(np.prod(indices.shape))
            width = int(np.prod(self.shape[1:]))
            raw_indices = indices._node.numpy_value().reshape(-1)
            if np.any(raw_indices < 0) or np.any(raw_indices >= rows):
                raise IndexError("gather index is outside the first dimension")
            encoded = np.zeros((count, rows), dtype=self.dtype)
            encoded[np.arange(count), raw_indices] = 1
            one_hot = Tensor(encoded, copy=False)
            source_grad = (one_hot.T @ grad.reshape(count, width)).reshape(
                self.shape
            )
            return ((self, source_grad),)

        result._backward = backward
        return result

    def sum(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        axes = _normalize_axes(axis, self.ndim)
        shape = _reduced_shape(self.shape, axes, keepdims)
        node = LazyNode(
            Op.REDUCE, (self._node,), ("sum", axes, keepdims), shape, self.dtype
        )
        tracks = _GRAD_ENABLED and self.requires_grad
        result = Tensor._from_node(node, tracks)
        if tracks:
            result._parents = (self,)

            def backward(grad: Tensor) -> BackwardResult:
                expanded = grad if keepdims else grad.reshape(
                    tuple(1 if index in axes else size for index, size in enumerate(self.shape))
                )
                return ((self, expanded * Tensor.ones(self.shape, dtype=self.dtype)),)

            result._backward = backward
        return result

    def mean(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        axes = _normalize_axes(axis, self.ndim)
        divisor = int(np.prod([self.shape[index] for index in axes]))
        return self.sum(axis=axes, keepdims=keepdims) / divisor

    def max(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        axes = _normalize_axes(axis, self.ndim)
        shape = _reduced_shape(self.shape, axes, keepdims)
        node = LazyNode(
            Op.REDUCE, (self._node,), ("max", axes, keepdims), shape, self.dtype
        )
        tracks = _GRAD_ENABLED and self.requires_grad
        result = Tensor._from_node(node, tracks)
        if tracks:
            result._parents = (self,)

            def backward(grad: Tensor) -> BackwardResult:
                expanded_shape = tuple(
                    1 if index in axes else size for index, size in enumerate(self.shape)
                )
                maximum = result if keepdims else result.reshape(expanded_shape)
                incoming = grad if keepdims else grad.reshape(expanded_shape)
                mask = self._binary(maximum, "equal")
                count = mask.sum(axis=axes, keepdims=True)
                return ((self, incoming * mask / count),)

            result._backward = backward
        return result

    def __repr__(self) -> str:
        state = "leaf" if self._is_leaf else "lazy"
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype}, "
            f"requires_grad={self.requires_grad}, {state})"
        )


def realize(
    *tensors: Tensor, compiler: Optional[str] = None
) -> Tuple[np.ndarray, ...]:
    return compile_graph(*tensors, compiler=compiler)()


def compile_graph(
    *tensors: Tensor,
    compiler: Optional[str] = None,
    dynamic_inputs: Sequence[Tensor] = (),
) -> CompiledProgram:
    if not tensors:
        raise ValueError("at least one tensor is required")
    selected = get_compiler(compiler)
    outputs = [tensor._node for tensor in tensors]
    if not dynamic_inputs:
        return selected.compile(outputs)
    return selected.compile(
        outputs, [tensor._node for tensor in dynamic_inputs]
    )
