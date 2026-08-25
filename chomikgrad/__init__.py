from .lazy import (
    Compiler,
    CompiledProgram,
    DeviceAdapter,
    LazyNode,
    NumpyCompiler,
    NumpyDeviceAdapter,
    Op,
    get_compiler,
    register_compiler,
    set_default_compiler,
)
from .mlx_backend import MLXCompiler, MLXDeviceAdapter, MLXProgram
from .coreml_backend import CoreMLCompiler, CoreMLDeviceAdapter, CoreMLProgram
from .cuda_backend import CUDACompiler, CUDADeviceAdapter, CUDAProgram
from .generation import GreedyVerification, verify_greedy_candidates
from .llama import (
    LlamaConfig,
    LlamaDecoderBlock,
    LlamaDecoderStep,
    LlamaForCausalLM,
)
from .nn import (
    CrossEntropyLoss,
    LayerNorm,
    Linear,
    Module,
    MultiHeadSelfAttention,
    Parameter,
    ReLU,
    Sequential,
    TransformerEncoderBlock,
    cross_entropy,
)
from .optim import SGD
from .tensor import Tensor, compile_graph, no_grad, realize

register_compiler("mlx", MLXCompiler)
register_compiler("coreml", CoreMLCompiler)
register_compiler("cuda", CUDACompiler)

__all__ = [
    "Compiler",
    "CompiledProgram",
    "CUDACompiler",
    "CUDADeviceAdapter",
    "CUDAProgram",
    "CoreMLCompiler",
    "CoreMLDeviceAdapter",
    "CoreMLProgram",
    "CrossEntropyLoss",
    "DeviceAdapter",
    "GreedyVerification",
    "LazyNode",
    "LayerNorm",
    "LlamaConfig",
    "LlamaDecoderBlock",
    "LlamaDecoderStep",
    "LlamaForCausalLM",
    "Linear",
    "Module",
    "MultiHeadSelfAttention",
    "MLXCompiler",
    "MLXDeviceAdapter",
    "MLXProgram",
    "NumpyCompiler",
    "NumpyDeviceAdapter",
    "Op",
    "Parameter",
    "ReLU",
    "SGD",
    "Sequential",
    "Tensor",
    "TransformerEncoderBlock",
    "compile_graph",
    "cross_entropy",
    "get_compiler",
    "no_grad",
    "realize",
    "register_compiler",
    "set_default_compiler",
    "verify_greedy_candidates",
]
