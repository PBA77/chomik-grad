from .lazy import (
    Compiler,
    CompiledProgram,
    LazyNode,
    NumpyCompiler,
    Op,
    get_compiler,
    register_compiler,
    set_default_compiler,
)
from .mlx_backend import MLXCompiler, MLXProgram
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

__all__ = [
    "Compiler",
    "CompiledProgram",
    "CrossEntropyLoss",
    "LazyNode",
    "LayerNorm",
    "Linear",
    "Module",
    "MultiHeadSelfAttention",
    "MLXCompiler",
    "MLXProgram",
    "NumpyCompiler",
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
]
