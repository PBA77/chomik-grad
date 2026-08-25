from __future__ import annotations

from typing import Iterable, List, Optional

from .lazy import get_compiler
from .nn import Parameter
from .tensor import Tensor, realize


class SGD:
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 0.01,
        *,
        inplace: bool = False,
    ) -> None:
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.parameters: List[Parameter] = list(parameters)
        self.lr = float(lr)
        self.inplace = bool(inplace)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self, compiler: Optional[str] = None) -> None:
        active = [parameter for parameter in self.parameters if parameter.grad is not None]
        if not active:
            return
        selected = get_compiler(compiler)
        parameters = [parameter._node for parameter in active]
        gradients = [
            parameter.grad._node
            for parameter in active
            if parameter.grad is not None
        ]
        native_update = (
            selected.update_parameters(
                parameters,
                gradients,
                self.lr,
                inplace=True,
            )
            if self.inplace
            else selected.update_parameters(parameters, gradients, self.lr)
        )
        if native_update is not None:
            parameter_nodes, gradient_nodes = native_update
            for parameter, parameter_node, gradient_node in zip(
                active, parameter_nodes, gradient_nodes
            ):
                parameter._node = parameter_node
                parameter.grad = Tensor._from_node(
                    gradient_node, requires_grad=False
                )
            return
        gradients = realize(
            *(parameter.grad for parameter in active if parameter.grad is not None),
            compiler=compiler,
        )
        for parameter, gradient in zip(active, gradients):
            if gradient.shape != parameter.shape:
                raise RuntimeError("gradient shape does not match parameter shape")
            storage = parameter._node.numpy_value()
            parameter.assign(storage - self.lr * gradient)
            # Keep an already evaluated gradient, matching ordinary optimizer
            # semantics if the caller deliberately does not zero it.
            parameter.grad = Tensor(gradient)
