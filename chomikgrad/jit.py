from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .lazy import Compiler, LazyNode, get_compiler
from .optim import SGD
from .tensor import Tensor


class CompiledTrainStep:
    """A fixed-shape forward/backward graph with dynamic batch inputs."""

    def __init__(
        self,
        loss_function: Callable[..., Tensor],
        optimizer: SGD,
        example_inputs: Sequence[Tensor],
        *,
        compiler: str,
        return_loss: bool,
    ) -> None:
        if not example_inputs:
            raise ValueError("compiled training requires at least one input")
        if any(not isinstance(value, Tensor) for value in example_inputs):
            raise TypeError("example inputs must be Tensor instances")

        self.compiler: Compiler = get_compiler(compiler)
        if (
            type(self.compiler).update_native_parameters
            is Compiler.update_native_parameters
        ):
            raise RuntimeError(
                f"the {self.compiler.device.name!r} compiler does not support "
                "compiled training steps"
            )
        self.optimizer = optimizer
        self.return_loss = bool(return_loss)
        self._input_placeholders = tuple(value._node for value in example_inputs)
        if len(set(self._input_placeholders)) != len(self._input_placeholders):
            raise ValueError("compiled training inputs must not alias each other")
        self._input_specs = tuple(
            (value.shape, value.dtype) for value in example_inputs
        )

        optimizer.zero_grad()
        loss = loss_function(*example_inputs)
        if not isinstance(loss, Tensor):
            raise TypeError("the compiled loss function must return a Tensor")
        if loss.shape != ():
            raise ValueError("the compiled loss function must return a scalar")
        loss.backward()

        active = tuple(
            parameter
            for parameter in optimizer.parameters
            if parameter.grad is not None
        )
        if not active:
            raise ValueError("the compiled loss has no trainable parameters")
        self._parameters = active
        self._parameter_placeholders = tuple(
            parameter._node for parameter in active
        )
        gradient_nodes = tuple(
            parameter.grad._node
            for parameter in active
            if parameter.grad is not None
        )
        dynamic_nodes = self._input_placeholders + self._parameter_placeholders
        if len(set(dynamic_nodes)) != len(dynamic_nodes):
            raise ValueError("compiled inputs and parameters must not alias")
        outputs = ((loss._node,) if self.return_loss else ()) + gradient_nodes
        self.program = self.compiler.compile(
            outputs,
            dynamic_inputs=dynamic_nodes,
        )
        self._loss_shape = loss.shape
        self._loss_dtype = loss.dtype
        optimizer.zero_grad()

    def _native_input(
        self,
        value: object,
        shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> object:
        if isinstance(value, Tensor):
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(
                    f"expected input shaped {shape} with dtype {dtype}, "
                    f"got {value.shape} with dtype {value.dtype}"
                )
            node = value._node
            native = node.native_values.get(self.compiler.device.name)
            if native is not None:
                return native
            array = node.numpy_value()
        else:
            array = np.asarray(value, dtype=dtype)
            if array.shape != shape:
                raise ValueError(f"expected input shaped {shape}, got {array.shape}")
        return self.compiler.device.array(array)

    def _native_parameter(self, node: LazyNode) -> object:
        backend = self.compiler.device.name
        native = node.native_values.get(backend)
        if native is None:
            native = self.compiler.device.array(node.numpy_value())
            if node.cache_native:
                node.native_values[backend] = native
        return native

    def __call__(self, *inputs: object) -> Optional[Tensor]:
        if len(inputs) != len(self._input_specs):
            raise ValueError(
                f"expected {len(self._input_specs)} inputs, got {len(inputs)}"
            )
        native_inputs = tuple(
            self._native_input(value, shape, dtype)
            for value, (shape, dtype) in zip(inputs, self._input_specs)
        )
        native_parameters = tuple(
            self._native_parameter(parameter._node)
            for parameter in self._parameters
        )
        bindings = dict(
            zip(
                self._input_placeholders + self._parameter_placeholders,
                native_inputs + native_parameters,
            )
        )
        outputs = self.program.run(bindings, synchronize=False)
        if self.return_loss:
            loss, *gradients = outputs
        else:
            loss = None
            gradients = list(outputs)
        parameter_nodes, gradient_nodes = self.compiler.update_native_parameters(
            tuple(parameter._node for parameter in self._parameters),
            gradients,
            self.optimizer.lr,
            inplace=self.optimizer.inplace,
        )
        for parameter, parameter_node, gradient_node in zip(
            self._parameters, parameter_nodes, gradient_nodes
        ):
            parameter._node = parameter_node
            parameter.grad = Tensor._from_node(
                gradient_node, requires_grad=False
            )
        if loss is None:
            return None
        return Tensor.from_native(
            loss,
            backend=self.compiler.device.name,
            shape=self._loss_shape,
            dtype=self._loss_dtype,
        )


def compile_train_step(
    loss_function: Callable[..., Tensor],
    optimizer: SGD,
    *example_inputs: Tensor,
    compiler: str,
    return_loss: bool = False,
) -> CompiledTrainStep:
    """Compile one fixed-shape training step for repeated execution."""

    return CompiledTrainStep(
        loss_function,
        optimizer,
        example_inputs,
        compiler=compiler,
        return_loss=return_loss,
    )
