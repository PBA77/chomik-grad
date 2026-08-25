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

    _SGD_FUSION_GROUP_SIZE = 16

    _SUPPORTED_LOWERINGS = {
        "elementwise_fusion",
        "layer_norm",
        "layer_norm_backward",
        "log_softmax",
        "log_softmax_backward",
        "softmax",
        "softmax_backward",
    }
    _LAYER_NORM_SOURCE = r"""
extern "C" __global__ void layer_norm(
    const float* inputs,
    const float* weight,
    const float* bias,
    float* outputs,
    const int width,
    const float epsilon
) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    const int offset = row * width;
    __shared__ float values[256];

    float sum = 0.0f;
    for (int column = thread; column < width; column += blockDim.x) {
        sum += inputs[offset + column];
    }
    values[thread] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) values[thread] += values[thread + stride];
        __syncthreads();
    }
    const float mean = values[0] / width;
    __syncthreads();

    float squared_sum = 0.0f;
    for (int column = thread; column < width; column += blockDim.x) {
        const float centered = inputs[offset + column] - mean;
        squared_sum += centered * centered;
    }
    values[thread] = squared_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) values[thread] += values[thread + stride];
        __syncthreads();
    }
    const float inverse_std = rsqrtf(values[0] / width + epsilon);

    for (int column = thread; column < width; column += blockDim.x) {
        outputs[offset + column] =
            (inputs[offset + column] - mean) * inverse_std * weight[column]
            + bias[column];
    }
}
"""
    _SOFTMAX_BACKWARD_SOURCE = r"""
extern "C" __global__ void softmax_backward(
    const float* gradients,
    const float* outputs,
    float* input_gradients,
    const int width
) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    const int offset = row * width;
    __shared__ float values[256];

    float projection = 0.0f;
    for (int column = thread; column < width; column += blockDim.x) {
        projection += gradients[offset + column] * outputs[offset + column];
    }
    values[thread] = projection;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) values[thread] += values[thread + stride];
        __syncthreads();
    }
    projection = values[0];

    for (int column = thread; column < width; column += blockDim.x) {
        const int index = offset + column;
        input_gradients[index] =
            outputs[index] * (gradients[index] - projection);
    }
}
"""
    _SOFTMAX_SOURCE = r"""
extern "C" __global__ void softmax_forward(
    const float* inputs,
    float* outputs,
    const int width,
    const int logarithmic
) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    const int offset = row * width;
    __shared__ float values[256];

    float maximum = -3.402823466e+38F;
    for (int column = thread; column < width; column += blockDim.x) {
        maximum = fmaxf(maximum, inputs[offset + column]);
    }
    values[thread] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            values[thread] = fmaxf(values[thread], values[thread + stride]);
        }
        __syncthreads();
    }
    maximum = values[0];

    float total = 0.0f;
    for (int column = thread; column < width; column += blockDim.x) {
        total += expf(inputs[offset + column] - maximum);
    }
    values[thread] = total;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) values[thread] += values[thread + stride];
        __syncthreads();
    }
    total = values[0];

    for (int column = thread; column < width; column += blockDim.x) {
        const int index = offset + column;
        const float shifted = inputs[index] - maximum;
        outputs[index] = logarithmic
            ? shifted - logf(total)
            : expf(shifted) / total;
    }
}
"""
    _LOG_SOFTMAX_BACKWARD_SOURCE = r"""
extern "C" __global__ void log_softmax_backward(
    const float* gradients,
    const float* outputs,
    float* input_gradients,
    const int width
) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    const int offset = row * width;
    __shared__ float values[256];

    float projection = 0.0f;
    for (int column = thread; column < width; column += blockDim.x) {
        projection += gradients[offset + column];
    }
    values[thread] = projection;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) values[thread] += values[thread + stride];
        __syncthreads();
    }
    projection = values[0];

    for (int column = thread; column < width; column += blockDim.x) {
        const int index = offset + column;
        input_gradients[index] =
            gradients[index] - expf(outputs[index]) * projection;
    }
}
"""
    _LAYER_NORM_BACKWARD_SOURCE = r"""
extern "C" __global__ void layer_norm_backward(
    const float* gradients,
    const float* inputs,
    const float* weight,
    float* input_gradients,
    float* weight_gradients,
    float* bias_gradients,
    const int width,
    const float epsilon
) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    const int offset = row * width;
    __shared__ float values[256];

    float sum = 0.0f;
    for (int column = thread; column < width; column += blockDim.x) {
        sum += inputs[offset + column];
    }
    values[thread] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) values[thread] += values[thread + stride];
        __syncthreads();
    }
    const float mean = values[0] / width;
    __syncthreads();

    float squared_sum = 0.0f;
    for (int column = thread; column < width; column += blockDim.x) {
        const float centered = inputs[offset + column] - mean;
        squared_sum += centered * centered;
    }
    values[thread] = squared_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) values[thread] += values[thread + stride];
        __syncthreads();
    }
    const float inverse_std = rsqrtf(values[0] / width + epsilon);
    __syncthreads();

    float weighted_sum = 0.0f;
    float correlation_sum = 0.0f;
    for (int column = thread; column < width; column += blockDim.x) {
        const int index = offset + column;
        const float normalized = (inputs[index] - mean) * inverse_std;
        const float weighted = gradients[index] * weight[column];
        weighted_sum += weighted;
        correlation_sum += weighted * normalized;
    }
    values[thread] = weighted_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) values[thread] += values[thread + stride];
        __syncthreads();
    }
    const float weighted_mean = values[0] / width;
    __syncthreads();

    values[thread] = correlation_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) values[thread] += values[thread + stride];
        __syncthreads();
    }
    const float correlation_mean = values[0] / width;

    for (int column = thread; column < width; column += blockDim.x) {
        const int index = offset + column;
        const float normalized = (inputs[index] - mean) * inverse_std;
        const float gradient = gradients[index];
        const float weighted = gradient * weight[column];
        input_gradients[index] = (
            weighted - weighted_mean - normalized * correlation_mean
        ) * inverse_std;
        atomicAdd(weight_gradients + column, gradient * normalized);
        atomicAdd(bias_gradients + column, gradient);
    }
}
"""

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
    _FUSED_BINARY = {
        "add": "left_value + right_value",
        "sub": "left_value - right_value",
        "mul": "left_value * right_value",
        "div": "left_value / right_value",
        "equal": "left_value == right_value ? 1.0f : 0.0f",
    }
    _FUSED_UNARY = {
        "identity": "input_value",
        "neg": "-input_value",
        "exp": "expf(input_value)",
        "log": "logf(input_value)",
        "sqrt": "sqrtf(input_value)",
        "relu": "fmaxf(input_value, 0.0f)",
        "step": "input_value > 0.0f ? 1.0f : 0.0f",
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
        self._layer_norm_kernel: Optional[Any] = None
        self._layer_norm_backward_kernel: Optional[Any] = None
        self._log_softmax_backward_kernel: Optional[Any] = None
        self._softmax_kernel: Optional[Any] = None
        self._softmax_backward_kernel: Optional[Any] = None
        self._elementwise_kernels: Dict[object, Any] = {}
        self._sgd_update_kernel: Optional[Any] = None
        self._sgd_group_kernels: Dict[object, Any] = {}

    @property
    def cache_size(self) -> int:
        return len(self._program_cache)

    def close(self) -> None:
        self.device.synchronize()
        self._program_cache.clear()
        self._elementwise_kernels.clear()
        self._sgd_group_kernels.clear()

    def _load_input(self, node: LazyNode) -> Any:
        if "cuda" in node.native_values:
            return node.native_values["cuda"]
        native = self.device.array(node.numpy_value())
        if node.cache_native:
            node.native_values["cuda"] = native
        return native

    @classmethod
    def _node_inputs(cls, node: LazyNode) -> Tuple[LazyNode, ...]:
        if node.lowering is not None and node.lowering[0] in cls._SUPPORTED_LOWERINGS:
            return node.lowering[1]
        return node.inputs

    @classmethod
    def _topological_sort(cls, outputs: Sequence[LazyNode]) -> Tuple[LazyNode, ...]:
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
    ) -> CUDAProgram:
        if not outputs:
            raise ValueError("at least one output is required")
        if any(node.op is not None for node in dynamic_inputs):
            raise ValueError("dynamic inputs must be graph leaves")

        nodes = self._topological_sort(outputs)
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
            compile(source, "<chomikgrad-cuda>", "exec"),
            {
                "cp": self._cp,
                "fused_elementwise": self._fused_elementwise,
                "layer_norm": self._layer_norm,
                "layer_norm_backward": self._layer_norm_backward,
                "log_softmax_backward": self._log_softmax_backward,
                "softmax": self._softmax,
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
        cp = self._cp
        can_fuse = all(
            parameter.dtype == cp.float32
            and gradient.dtype == cp.float32
            and parameter.flags.c_contiguous
            and gradient.flags.c_contiguous
            and parameter.shape == gradient.shape
            for parameter, gradient in zip(native_parameters, gradients)
        )
        if can_fuse:
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
        else:
            updated = tuple(
                self._update_parameter(
                    parameter,
                    gradient,
                    learning_rate,
                    inplace=inplace,
                )
                for parameter, gradient in zip(native_parameters, gradients)
            )
        parameter_nodes = tuple(
            LazyNode.native_leaf("cuda", value, node.shape, node.dtype)
            for node, value in zip(parameters, updated)
        )
        gradient_nodes = tuple(
            LazyNode.native_leaf("cuda", value, node.shape, node.dtype)
            for node, value in zip(parameters, gradients)
        )
        return parameter_nodes, gradient_nodes

    def _update_parameter_group(
        self,
        parameters: Sequence[Any],
        gradients: Sequence[object],
        learning_rate: float,
        *,
        inplace: bool,
    ) -> Tuple[Any, ...]:
        cp = self._cp
        outputs = (
            tuple(parameters)
            if inplace
            else tuple(cp.empty_like(parameter) for parameter in parameters)
        )
        sizes = tuple(int(parameter.size) for parameter in parameters)
        total = sum(sizes)
        if total == 0:
            return outputs

        key = (sizes, inplace)
        kernel = self._sgd_group_kernels.get(key)
        if kernel is None:
            declarations = []
            branches = []
            offset = 0
            for index, size in enumerate(sizes):
                declarations.extend(
                    (
                        f"    {'float' if inplace else 'const float'}* "
                        f"parameter_{index}",
                        f"    const float* gradient_{index}",
                    )
                )
                output_name = f"parameter_{index}"
                if not inplace:
                    declarations.append(f"    float* output_{index}")
                    output_name = f"output_{index}"
                end = offset + size
                branches.append(
                    f"    if (gid < {end}ULL) {{\n"
                    f"        const unsigned long long item = gid - {offset}ULL;\n"
                    f"        {output_name}[item] = parameter_{index}[item] - "
                    f"learning_rate * gradient_{index}[item];\n"
                    f"        return;\n"
                    f"    }}"
                )
                offset = end
            declarations.append("    const float learning_rate")
            source = f"""
extern "C" __global__ void sgd_update_group(
{',\n'.join(declarations)}
) {{
    const unsigned long long gid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
{chr(10).join(branches)}
}}
"""
            kernel = cp.RawKernel(source, "sgd_update_group")
            self._sgd_group_kernels[key] = kernel

        arguments = []
        for parameter, gradient, output in zip(parameters, gradients, outputs):
            arguments.extend((parameter, gradient))
            if not inplace:
                arguments.append(output)
        arguments.append(np.float32(learning_rate))
        kernel(((total + 255) // 256,), (256,), tuple(arguments))
        return outputs

    def _update_parameter(
        self,
        parameter: Any,
        gradient: Any,
        learning_rate: float,
        *,
        inplace: bool = False,
    ) -> Any:
        cp = self._cp
        if self._sgd_update_kernel is None:
            self._sgd_update_kernel = cp.ElementwiseKernel(
                "T parameter, T gradient, T learning_rate",
                "T updated",
                "updated = parameter - learning_rate * gradient",
                "chomik_sgd_update",
            )
        arguments = (
            parameter,
            gradient,
            parameter.dtype.type(learning_rate),
        )
        if inplace:
            self._sgd_update_kernel(*arguments, parameter)
            return parameter
        return self._sgd_update_kernel(*arguments)

    def _expression(self, node: LazyNode, args: Sequence[str]) -> str:
        if node.lowering is not None and node.lowering[0] in self._SUPPORTED_LOWERINGS:
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
            raise ValueError(f"unsupported CUDA lowering: {kind}")

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
            left, right = node.inputs
            if len(left.shape) > 2 and len(right.shape) == 2:
                flattened = (-1, left.shape[-1])
                return (
                    f"cp.matmul(cp.reshape({args[0]}, {flattened!r}), {args[1]})"
                    f".reshape({node.shape!r})"
                )
            if (
                len(left.shape) > 2
                and len(right.shape) > 2
                and int(np.prod(node.shape[:-2])) == 1
            ):
                left_matrix = (left.shape[-2], left.shape[-1])
                right_matrix = (right.shape[-2], right.shape[-1])
                return (
                    f"cp.matmul(cp.reshape({args[0]}, {left_matrix!r}), "
                    f"cp.reshape({args[1]}, {right_matrix!r}))"
                    f".reshape({node.shape!r})"
                )
            return f"cp.matmul({args[0]}, {args[1]})"
        if node.op is Op.GATHER:
            return f"cp.take({args[0]}, {args[1]}, axis=0)"
        raise ValueError(f"unsupported operation: {node.op}")

    def _fused_elementwise(
        self,
        expression: object,
        inputs: Sequence[Any],
        output_shape: Tuple[int, ...],
    ) -> Any:
        cp = self._cp
        if any(value.dtype != cp.float32 for value in inputs):
            raise TypeError("CUDA fused elementwise operations require float32")

        def render(node: object) -> str:
            parts = tuple(node)  # type: ignore[arg-type]
            kind = str(parts[0])
            if kind == "input":
                return f"input_{int(parts[1])}"
            if kind in self._FUSED_BINARY:
                left = render(parts[1])
                right = render(parts[2])
                return (
                    self._FUSED_BINARY[kind]
                    .replace("left_value", f"({left})")
                    .replace("right_value", f"({right})")
                )
            if kind in self._FUSED_UNARY:
                value = render(parts[1])
                return self._FUSED_UNARY[kind].replace(
                    "input_value", f"({value})"
                )
            raise ValueError(f"unsupported fused elementwise operation: {kind}")

        key = (expression, len(inputs))
        kernel = self._elementwise_kernels.get(key)
        if kernel is None:
            input_parameters = ", ".join(
                f"float32 input_{index}" for index in range(len(inputs))
            )
            kernel = cp.ElementwiseKernel(
                input_parameters,
                "float32 output",
                f"output = {render(expression)}",
                f"chomik_fused_elementwise_{len(self._elementwise_kernels)}",
            )
            self._elementwise_kernels[key] = kernel
        output = cp.empty(output_shape, dtype=cp.float32)
        kernel(*inputs, output)
        return output

    def _layer_norm(
        self,
        inputs: Any,
        weight: Any,
        bias: Any,
        epsilon: float,
    ) -> Any:
        cp = self._cp
        if (
            inputs.dtype != cp.float32
            or weight.dtype != cp.float32
            or bias.dtype != cp.float32
            or not inputs.flags.c_contiguous
            or not weight.flags.c_contiguous
            or not bias.flags.c_contiguous
        ):
            mean = cp.mean(inputs, axis=-1, keepdims=True)
            centered = inputs - mean
            variance = cp.mean(centered * centered, axis=-1, keepdims=True)
            return centered / cp.sqrt(variance + epsilon) * weight + bias

        if self._layer_norm_kernel is None:
            self._layer_norm_kernel = cp.RawKernel(
                self._LAYER_NORM_SOURCE,
                "layer_norm",
            )
        output = cp.empty_like(inputs)
        width = inputs.shape[-1]
        rows = inputs.size // width
        self._layer_norm_kernel(
            (rows,),
            (256,),
            (inputs, weight, bias, output, np.int32(width), np.float32(epsilon)),
        )
        return output

    def _softmax(self, inputs: Any, axis: int, logarithmic: bool) -> Any:
        cp = self._cp
        if inputs.size == 0:
            return cp.empty_like(inputs)
        if (
            axis != inputs.ndim - 1
            or inputs.dtype != cp.float32
            or not inputs.flags.c_contiguous
        ):
            maximum = cp.max(inputs, axis=axis, keepdims=True)
            shifted = inputs - maximum
            exponentials = cp.exp(shifted)
            total = cp.sum(exponentials, axis=axis, keepdims=True)
            return shifted - cp.log(total) if logarithmic else exponentials / total

        if self._softmax_kernel is None:
            self._softmax_kernel = cp.RawKernel(
                self._SOFTMAX_SOURCE,
                "softmax_forward",
            )
        output = cp.empty_like(inputs)
        width = inputs.shape[-1]
        rows = inputs.size // width
        self._softmax_kernel(
            (rows,),
            (256,),
            (inputs, output, np.int32(width), np.int32(logarithmic)),
        )
        return output

    def _log_softmax_backward(
        self,
        gradients: Any,
        outputs: Any,
        axis: int,
    ) -> Any:
        cp = self._cp
        if gradients.size == 0:
            return cp.empty_like(gradients)
        if (
            axis != gradients.ndim - 1
            or gradients.dtype != cp.float32
            or outputs.dtype != cp.float32
            or not gradients.flags.c_contiguous
            or not outputs.flags.c_contiguous
        ):
            projection = cp.sum(gradients, axis=axis, keepdims=True)
            return gradients - cp.exp(outputs) * projection

        if self._log_softmax_backward_kernel is None:
            self._log_softmax_backward_kernel = cp.RawKernel(
                self._LOG_SOFTMAX_BACKWARD_SOURCE,
                "log_softmax_backward",
            )
        input_gradients = cp.empty_like(gradients)
        width = gradients.shape[-1]
        rows = gradients.size // width
        self._log_softmax_backward_kernel(
            (rows,),
            (256,),
            (gradients, outputs, input_gradients, np.int32(width)),
        )
        return input_gradients

    def _softmax_backward(
        self,
        gradients: Any,
        outputs: Any,
        axis: int,
    ) -> Any:
        cp = self._cp
        if (
            axis != gradients.ndim - 1
            or gradients.dtype != cp.float32
            or outputs.dtype != cp.float32
            or not gradients.flags.c_contiguous
            or not outputs.flags.c_contiguous
            or gradients.size == 0
        ):
            weighted = gradients * outputs
            projection = cp.sum(weighted, axis=axis, keepdims=True)
            return outputs * (gradients - projection)

        if self._softmax_backward_kernel is None:
            self._softmax_backward_kernel = cp.RawKernel(
                self._SOFTMAX_BACKWARD_SOURCE,
                "softmax_backward",
            )
        input_gradients = cp.empty_like(gradients)
        width = gradients.shape[-1]
        rows = gradients.size // width
        self._softmax_backward_kernel(
            (rows,),
            (256,),
            (gradients, outputs, input_gradients, np.int32(width)),
        )
        return input_gradients

    def _layer_norm_backward(
        self,
        gradients: Any,
        inputs: Any,
        weight: Any,
        epsilon: float,
    ) -> Tuple[Any, Any, Any]:
        cp = self._cp
        if (
            gradients.dtype != cp.float32
            or inputs.dtype != cp.float32
            or weight.dtype != cp.float32
            or not gradients.flags.c_contiguous
            or not inputs.flags.c_contiguous
            or not weight.flags.c_contiguous
            or inputs.size == 0
        ):
            mean = cp.mean(inputs, axis=-1, keepdims=True)
            centered = inputs - mean
            variance = cp.mean(centered * centered, axis=-1, keepdims=True)
            inverse_std = 1.0 / cp.sqrt(variance + epsilon)
            normalized = centered * inverse_std
            weighted = gradients * weight
            input_gradients = (
                weighted
                - cp.mean(weighted, axis=-1, keepdims=True)
                - normalized
                * cp.mean(weighted * normalized, axis=-1, keepdims=True)
            ) * inverse_std
            rows = gradients.reshape((-1, gradients.shape[-1]))
            normalized_rows = normalized.reshape(rows.shape)
            weight_gradients = cp.sum(rows * normalized_rows, axis=0)
            bias_gradients = cp.sum(rows, axis=0)
            return input_gradients, weight_gradients, bias_gradients

        if self._layer_norm_backward_kernel is None:
            self._layer_norm_backward_kernel = cp.RawKernel(
                self._LAYER_NORM_BACKWARD_SOURCE,
                "layer_norm_backward",
            )
        input_gradients = cp.empty_like(inputs)
        weight_gradients = cp.zeros_like(weight)
        bias_gradients = cp.zeros_like(weight)
        width = inputs.shape[-1]
        rows = inputs.size // width
        self._layer_norm_backward_kernel(
            (rows,),
            (256,),
            (
                gradients,
                inputs,
                weight,
                input_gradients,
                weight_gradients,
                bias_gradients,
                np.int32(width),
                np.float32(epsilon),
            ),
        )
        return input_gradients, weight_gradients, bias_gradients
