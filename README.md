# chomik-grad

Minimal tensor framework in Python: lazy graphs, autograd, pluggable compilers,
and a small set of neural-network training tools. The runtime requires only
NumPy.

## Architecture

The entire IR has six operations:

1. `ELEMENTWISE` — add, multiply, exp, log, sqrt, and ReLU as variants of one operation,
2. `REDUCE` — sum/max,
3. `RESHAPE`,
4. `PERMUTE`,
5. `MATMUL`,
6. `GATHER` — indexing along the first axis, used by embeddings among other things.

`softmax` and `log_softmax` are numerically stable compositions of these
primitives. They do not add another instruction or a special case to backends.

Operations on `Tensor` only build a graph. `numpy()`, `item()`, `realize()`, or
`SGD.step()` pass the complete required graph to the selected compiler. The
default `cpu` plugin generates a straight-line Python function containing NumPy
calls and then executes it.

CUDA and OpenCL first build a backend-local `GraphPlan`. The plan selects
supported lowerings and fusions without writing them to `LazyNode`, so compiling
with one backend cannot change the graph later used by another. Both backends
share graph analysis and a straight-line Python program generator; only their
mappings from operations to native kernels differ.

```python
import numpy as np
from chomikgrad import Linear, SGD, Tensor, cross_entropy

model = Linear(4, 3)
optimizer = SGD(model.parameters(), lr=0.1)
x = Tensor(np.random.randn(8, 4).astype(np.float32))
y = np.array([0, 1, 2, 0, 1, 2, 0, 1])

loss = cross_entropy(model(x), y)  # nothing has been computed yet
loss.backward()                    # builds the lazy gradient graph
optimizer.step()                   # compiles and executes it on the CPU
```

CUDA and OpenCL SGD can optionally update existing parameter storage:

```python
optimizer = SGD(model.parameters(), lr=0.1, inplace=True)
optimizer.step(compiler="cuda")
```

The default `inplace=False` preserves the weight snapshot used by previously
built lazy graphs. In-place mode reduces peak memory, but those old graphs then
observe the updated weights.

By default, `Tensor(np_array)` owns a copy of its data. For fresh or immutable
arrays, `Tensor(np_array, copy=False)` can be used explicitly to avoid an
additional RAM copy. The MLX backend does not cache such an input and therefore
observes later changes to the source array.

## Compiler and device plugins

A compiler receives output `LazyNode` objects and returns a `CompiledProgram`.
Every program exposes the same `run(bindings, synchronize=False)` contract, so
leaf values can be replaced without rebuilding the graph:

```python
native_outputs = program.run(
    {input_node: program.device.array(new_value)},
    synchronize=False,
)
```

For inference, callers can additionally identify which leaves actually change
between invocations. The backend can then capture weights and other constants:

```python
program = compile_graph(
    output,
    compiler="mlx",
    dynamic_inputs=(tokens, position, key_cache, value_cache),
)
```

MLX uses this information only to specialize inference programs. Regular
`compile_graph(...)`, autograd, and training still pass every leaf dynamically,
so parameter updates do not require recompilation.

A plugin consists of a few small components:

```python
from chomikgrad import Compiler, DeviceAdapter, register_compiler

class MyDevice(DeviceAdapter):
    # array, evaluate, synchronize, to_numpy, argmax, and dtype
    ...

class MyCompiler(Compiler):
    device = MyDevice()

    def compile(self, outputs):
        # Traverse each LazyNode's inputs and handle all six Op values.
        # Return a CompiledProgram using the same device.
        ...

register_compiler("my-device", MyCompiler)
```

`DeviceAdapter` separates native array creation, synchronization, NumPy
readback, `argmax`, dtype mapping, and optional safetensors loading from IR
compilation. Additional backends can therefore reuse the same inference
runtime, although each must still provide its own six lowerings and any fast
RMSNorm, RoPE, or attention kernels. A compiler can be selected for a single
realization:
`tensor.numpy(compiler="my-device")`.

## Apple Silicon GPU through MLX

The optional `mlx` plugin translates the same six-operation IR to `mlx.core`
and explicitly executes the graph on a Metal GPU device. Missing MLX or Metal
support produces a clear error; the backend never silently falls back to CPU.

Parameters and gradients remain native MLX arrays on the GPU between steps.
Structurally identical graphs reuse the cache and `mx.compile`, while SGD
computes gradients and updates parameters with a single GPU synchronization and
without copying them through NumPy. Data is transferred to RAM only after an
explicit `numpy()`, `item()`, or use of the `cpu` compiler. MLX implements the
common `CompiledProgram.run(...)` and `DeviceAdapter` interfaces instead of
requiring a generator-specific API. This allows one autoregressive decode
program to be retained while binding new tokens and KV caches to it. Once a
fast lowering has been selected, the compiler prunes its unused portable
expansion; the complete fallback remains available to CPU and other backends.

MLX requires Apple Silicon, macOS 14+, and a native Python 3.10+. Example setup
when the system Python is older:

```bash
/opt/homebrew/bin/python3 -m venv .venv
.venv/bin/python -m pip install '.[demo,mlx]'
.venv/bin/python examples/train_digits.py --compiler mlx
```

## NVIDIA GPUs through CUDA

The optional `cuda` plugin executes the same six-operation IR through CuPy. It
never silently falls back to CPU, and SGD parameters and gradients stay in GPU
memory between steps. The `ctk` dependency variant bundles the required CUDA
components, so a compatible NVIDIA driver is sufficient:

```bash
python -m pip install '.[benchmark,cuda]'
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python benchmarks/compare_tinygrad_10_cases.py --device cuda --trials 3
python benchmarks/compare_tinygrad_10_cases.py --device cuda --trials 5 --chomik-jit
```

The optional `compile_train_step` captures a fixed-shape forward and backward
graph only once. CUDA fuses elementwise chains, softmax/log-softmax, and SGD
updates in groups of up to 16 parameters. This replays a generated Chomik
program rather than a CUDA Graph: CuPy 14.2 cannot currently capture the
`cp.matmul` calls used here because setting the cuBLAS stream during capture is
unsupported.

On native Windows, the `torch-compile` variant uses the official `cudagraphs`
backend because the PyTorch wheel does not include Triton. On systems with a
working Triton installation, full Inductor can be selected with
`--torch-compile-backend inductor`.

## GPUs through OpenCL

The optional `opencl` plugin executes the complete FP32 IR through PyOpenCL.
Elementwise operations, reductions, reshape, permute, gather, and SGD use
custom kernels, while regular, strided-batched, and offset-batched `MATMUL` use
CLBlast. LayerNorm, softmax, and log-softmax have fused forward and backward
kernels. Single-consumer elementwise chains of the same shape are fused without
duplicating computation. `reshape` and `permute` use lightweight runtime views,
and SGD updates are grouped in batches of up to 16 parameters per kernel.
Parameters and gradients remain on the GPU between steps, with no silent CPU
fallback.

PyOpenCL is installed through the project extra, while the shared CLBlast 1.7
library must be installed separately. Its path can be provided explicitly:

```powershell
python -m pip install '.[benchmark,opencl]'
$env:CLBLAST_PATH='C:\path\to\clblast.dll'
python benchmarks\compare_tinygrad_10_cases.py --device opencl --trials 3
```

A repeated training step with a fixed batch shape can capture forward,
backward, and SGD once:

```python
from chomikgrad import SGD, Tensor, compile_train_step

optimizer = SGD(model.parameters(), lr=0.03)

def loss_function(inputs, targets):
    logits = model(inputs)
    return -(logits.log_softmax(axis=1) * targets).sum() / inputs.shape[0]

step = compile_train_step(
    loss_function,
    optimizer,
    Tensor.zeros((64, 8, 8)),
    Tensor.zeros((64, 10)),
    compiler="opencl",
)
step(batch_inputs, batch_targets_one_hot)
```

Inputs may change values but must keep the examples' shapes and dtypes. A
shorter final batch requires a second compiled step. By default the step does
not materialize its loss; `return_loss=True` enables loss reporting.

On Linux, `CLBLAST_PATH` may point to `libclblast.so`. The backend also checks
the system loader and `Library/bin/clblast.dll` or `lib/libclblast.so` in the
active virtual environment. The benchmark selects tinygrad's `CL` device, so
both implementations use OpenCL on the same GPU.

The backend deliberately rejects dtypes other than FP32, except int32/int64
`GATHER` indices. The NVIDIA driver used for the measurements exposed OpenCL
3.0 but only OpenCL C 1.2 and no `cl_khr_fp16`, so the backend does not pretend
to provide FP16 by converting it to FP32.

On a GeForce RTX 5070 Ti (`PyOpenCL 2026.1.3`, `CLBlast 1.7.0`,
`tinygrad 0.14.0`, Python 3.14.2), one complete run with the default number of
microbenchmark repetitions produced:

| case | Chomik OpenCL | tinygrad CL |
|---|---:|---:|
| elementwise, 1M | **1.964 ms** | 2.886 ms |
| reduce sum, 4M | **1.327 ms** | 1.799 ms |
| softmax, 1024x1024 | **1.438 ms** | 2.874 ms |
| matmul, 64x64 | **0.277 ms** | 2.342 ms |
| matmul, 256x256 | **0.344 ms** | 2.393 ms |
| matmul, 1024x1024 | **2.370 ms** | 3.904 ms |
| matmul, 2048x2048 | **7.864 ms** | 9.556 ms |
| batched matmul, 16x4x64 | **0.734 ms** | 2.894 ms |
| MLP training, 20 epochs | 3.092 s | **2.262 s** |
| transformer training, 10 epochs | 7.776 s | **6.695 s** |

Fused backward kernels, buffer pooling, one-time invoker preparation,
offset-batched GEMM for partial broadcasting, and cached offset plans reduced
transformer training from 331.278 s to 7.776 s, a 42.6x improvement. Chomik won
all eight microbenchmarks; tinygrad remained 37% faster for MLP training and 16%
faster for transformer training. CUDA remains the preferred backend for maximum
performance on the same GPU.

A separate series of five complete training trials compared capture after
fusion with TinyJit. `--repeat-scale 0.01` shortened only the microbenchmarks
and did not change the epoch counts:

| variant | MLP first | MLP median | Transformer first | Transformer median |
|---|---:|---:|---:|---:|
| Chomik `compile_train_step` | **1.327 s** | **0.626 s** | **4.250 s** | **2.237 s** |
| tinygrad `TinyJit` | 3.336 s | 2.217 s | 8.957 s | 2.816 s |

Compared with capture before fusion, Chomik's median fell from 1.023 s to
0.626 s for the MLP and from 3.965 s to 2.237 s for the transformer. Softmax
and elementwise fusion removed 28 executable operations from the transformer
graph, lightweight views reduced Python overhead, and grouped SGD reduced the
39 weight-update kernels to three per step. After warm-up, Chomik is 3.54x
faster than tinygrad for the MLP and 1.26x faster for the transformer; the
transformer's first run is 52.5% shorter.

## GPUs through Vulkan

The optional `vulkan` plugin uses `wgpu-native` and selects only an adapter
whose `backend_type` is `Vulkan`; it cannot silently switch to D3D12 or OpenGL.
Each lazy graph is encoded as one command buffer containing WGSL kernels.
Reshape and permute preserve views through strides, while tiled matmul supports
batches and broadcasting without Python-side dispatch loops. The backend covers
the complete FP32 IR, autograd, and regular or in-place SGD.

`wgpu` requires Python 3.11 or newer:

```powershell
python -m pip install '.[benchmark,vulkan]'
python -m pip install 'dawn-python==0.3.0'
python benchmarks\compare_tinygrad_10_cases.py --device vulkan --micro-only
```

Comparison with tinygrad WEBGPU additionally requires `dawn-python==0.3.0`.
The script sets `WEBGPU_BACKEND=WGPUBackendType_Vulkan`, so Dawn also cannot
select another API. Int64 indices are range-checked and narrowed to int32
because WGSL has no portable int64 storage-buffer type.

On the same GeForce RTX 5070 Ti (`wgpu 0.31.1`, `dawn-python 0.3.0`,
`tinygrad 0.14.0`), the complete microbenchmark repetition set produced:

| case | Chomik Vulkan | tinygrad WEBGPU/Vulkan |
|---|---:|---:|
| elementwise, 1M | **5.111 ms** | 8.427 ms |
| reduce sum, 4M | **5.535 ms** | 42.152 ms |
| softmax, 1024x1024 | **3.755 ms** | 52.596 ms |
| matmul, 64x64 | **0.828 ms** | 3.347 ms |
| matmul, 256x256 | **0.929 ms** | 3.465 ms |
| matmul, 1024x1024 | **4.735 ms** | 9.807 ms |
| matmul, 2048x2048 | **20.517 ms** | 32.764 ms |
| batched matmul, 16x4x64 | **2.508 ms** | 4.721 ms |

Chomik won all eight cases. Its single complete worker ran 20 MLP epochs in
1.463 s and 10 transformer epochs in 5.911 s. The tinygrad/Dawn training worker
did not finish within five minutes and was stopped, so no misleading result is
reported. For transformer training, Chomik Vulkan was 1.32x faster than OpenCL
(7.776 s) but 2.8x slower than CUDA (2.095 s).

## Apple Silicon Neural Engine through Core ML

The optional `coreml` plugin compiles the same six-operation IR into an FP16
`ML Program` and loads it with `CPU_AND_NE`. Apple provides neither a direct ANE
compute API nor an `NE_ONLY` mode: Core ML makes the final decision separately
for every operation. `CoreMLProgram.compute_plan_summary()` reads the compiled
model's execution plan, so tests do not treat the `CPU_AND_NE` flag alone as
proof that the Neural Engine was used.

The backend is inference-only. It captures constant weights, recognizes
`x @ weight.T` as Core ML `linear`, and splits models with more than 1 GiB of
constants into segments. Without segmentation, monolithic TinyLlama 1.1B ran
correctly, but Core ML assigned the entire graph to the CPU on M1. Three
segments preserve identical output and move most of the graph back to the ANE.

```bash
.venv/bin/python -m pip install '.[coreml,llm]'
.venv/bin/python examples/generate_tinyllama.py \
  --compiler coreml --dtype float16 --max-new-tokens 8
```

The limitations are deliberately explicit:

- FP16 only, macOS 15+, and Apple Silicon,
- fixed-shape inference; no autograd or training,
- `MLModel.predict` is synchronous and crosses through NumPy arrays at segment
  boundaries,
- a single token at `batch=1` does not use the ANE as efficiently as the GPU.

The like-for-like comparison uses exactly the same weights, activations, KV
cache, Chomik implementation, prompt, and FP16 tokens:

```bash
.venv/bin/python benchmarks/tinyllama_coreml_vs_mlx.py --trials 3
```

Example result on an Apple M1 Max (`coremltools 9.0`, `MLX 0.32.1`, a 28-token
prompt, and eight response tokens):

| TinyLlama 1.1B FP16 | Core ML / ANE | MLX / GPU |
|---|---:|---:|
| first token, including compilation | 42.879 s | **0.061 s** |
| complete response, including compilation | 71.836 s | **0.140 s** |
| first token after warm-up | 0.793 s | **0.021 s** |
| warm decode | 15.0 tokens/s | **114.5 tokens/s** |
| complete response after warm-up | 1.424 s | **0.083 s** |

The Compute Plan reported 1,377 prefill operations preferring the Neural Engine
and 185 preferring the CPU; the decode counts were 1,235 and 218 respectively.
All eight generated token IDs remained identical. On M1, the ANE is therefore a
working research backend, but it is not performance-competitive with the Metal
GPU for an autoregressive LLM at `batch=1`.

Apple documentation: [selecting the CPU and Neural Engine](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine),
[Compute Plan](https://apple.github.io/coremltools/docs-guides/source/mlmodel-utilities.html),
and [FP16 execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html).

## Running the project

```bash
python -m unittest discover -s tests -v
python -m pip install '.[demo]'
python examples/train_digits.py
```

The test suite also checks backend-plan isolation: an elementwise fusion in a
CUDA or OpenCL plan must not change the portable graph or another backend's
plan. The latest complete verification ran 70 tests; 14 optional tests for
backends unavailable on this host were skipped.

The demo trains a `64 -> 48 -> 10` MLP on scikit-learn's bundled, freely
available 8x8 digits dataset. The script fails if test accuracy does not reach
90%.

## Transformer

`MATMUL` also supports batch dimensions, so the same six-operation IR covers
multi-head attention without a dedicated instruction. The package includes
`LayerNorm`, `MultiHeadSelfAttention`, and a pre-norm
`TransformerEncoderBlock`.

The second example treats the eight rows of a digit image as eight tokens. It
uses a 32-dimensional embedding, two encoder blocks, four heads, a 64-unit MLP,
and mean pooling:

```bash
python examples/train_digits_transformer.py --compiler cpu
.venv/bin/python examples/train_digits_transformer.py --compiler mlx
```

## Benchmark against tinygrad

The main benchmark runs Chomik, tinygrad, and optional PyTorch variants in
separate processes so their GPU runtimes do not affect one another. It covers
eight tensor operations, 20 MLP epochs, and 10 transformer epochs. It validates
result compatibility and accuracy without unstable timing thresholds:

```bash
.venv/bin/python -m pip install '.[benchmark]'
.venv/bin/python benchmarks/compare_tinygrad_10_cases.py --trials 3
.venv/bin/python benchmarks/compare_tinygrad_10_cases.py --json
```

Use `--device cuda` for NVIDIA/CUDA, `--device opencl` for OpenCL, or
`--device vulkan` for Vulkan. The default `metal` mode preserves the existing
Apple Silicon behavior. `--micro-only` skips the two longer training cases.
`--chomik-jit` captures both Chomik training workloads on CUDA or OpenCL.

`benchmarks/transformer_vs_tinygrad.py` remains available as a shorter,
transformer-only benchmark.

Example result from the main benchmark on an Apple M1 Max (`tinygrad 0.14.0`,
`mlx 0.32.1`):

| case | Chomik | tinygrad |
|---|---:|---:|
| elementwise, 1M | **0.75 ms** | 1.82 ms |
| reduce sum, 4M | **0.74 ms** | 1.60 ms |
| softmax, 1024x1024 | **0.65 ms** | 1.95 ms |
| matmul, 64x64 | **0.29 ms** | 2.41 ms |
| matmul, 256x256 | **0.33 ms** | 2.46 ms |
| matmul, 1024x1024 | **1.57 ms** | 3.24 ms |
| matmul, 2048x2048 | **5.35 ms** | 6.59 ms |
| batched matmul, 16x4x64 | **0.40 ms** | 2.67 ms |
| MLP training, 20 epochs | **0.37 s** | 1.06 s |
| transformer training, 10 epochs | **1.46 s** | 1.63 s |

This is a small model, so the result also measures Python and compilation
overhead. Ratios may differ with other library versions and Apple chips.

On an NVIDIA GeForce RTX 5070 Ti (`CuPy 14.2.0`, `tinygrad 0.14.0`,
`PyTorch 2.13.0+cu130`, Python 3.14.2), a complete
`--device cuda --trials 3` run produced these medians:

| case | Chomik CUDA | tinygrad CUDA | PyTorch eager | PyTorch compile/CUDA Graphs |
|---|---:|---:|---:|---:|
| elementwise, 1M | 1.554 ms | 2.767 ms | **0.971 ms** | 1.241 ms |
| reduce sum, 4M | 1.295 ms | 1.965 ms | **0.925 ms** | 1.422 ms |
| softmax, 1024x1024 | 1.248 ms | 2.689 ms | **0.656 ms** | 0.825 ms |
| matmul, 64x64 | 0.116 ms | 2.301 ms | **0.108 ms** | 0.187 ms |
| matmul, 256x256 | 0.215 ms | 2.385 ms | **0.186 ms** | 0.246 ms |
| matmul, 1024x1024 | 1.607 ms | 3.876 ms | **0.980 ms** | 1.281 ms |
| matmul, 2048x2048 | 6.689 ms | 9.948 ms | **3.981 ms** | 4.362 ms |
| batched matmul, 16x4x64 | 0.667 ms | 2.921 ms | **0.367 ms** | 0.541 ms |
| MLP training, 20 epochs | 0.563 s | 1.233 s | **0.253 s** | 0.375 s |
| transformer training, 10 epochs | 1.682 s | 1.722 s | **1.007 s** | 1.718 s |

PyTorch eager won all ten cases. Fingerprint and accuracy checks passed for
every framework. The microbenchmarks include NumPy-to-GPU input transfer and
NumPy result readback. Fused softmax and LayerNorm backward reduced transformer
training time from 2.257 s to 1.682 s.

After adding CUDA JIT, five alternating eager/JIT trials were run in one
process. The current implementation produced these medians:

| training workload | Chomik CUDA | Chomik CUDA JIT | speedup |
|---|---:|---:|---:|
| MLP, 20 epochs | 0.427 s | **0.233 s** | 1.83x |
| transformer, 10 epochs | 1.640 s | **0.987 s** | 1.66x |

Accuracy remained within its previous range. These numbers are a separate A/B
series and should not be mixed with the earlier four-framework table, which was
measured under a different GPU load.

### Inference of an approximately 1B-parameter LLM core

The second benchmark builds a decoder-only transformer core without an
embedding, tokenizer, or LM head. Its default configuration has 20 blocks,
width 2,048, 16 heads, an 8,192-unit FFN, sequence length 32, and exactly
1,007,169,536 parameters:

```bash
.venv/bin/python benchmarks/llm_1b_inference.py
.venv/bin/python benchmarks/llm_1b_inference.py --json
```

Run the same model on NVIDIA/CUDA with
`python benchmarks/llm_1b_inference.py --device cuda`.

On an M1 Max with FP32 and `batch=1`, the median of ten warm forwards was
32.83 ms for Chomik and 40.38 ms for tinygrad. Their first forwards took 0.62 s
and 2.63 s respectively. This is a prefill of synthetic hidden states, not
autoregressive generation with a KV cache.

On an RTX 5070 Ti, the same FP32 benchmark
(`--device cuda --warm-runs 30`) produced matching output fingerprints and the
following results:

| metric | Chomik CUDA | tinygrad CUDA | PyTorch eager | PyTorch compile/CUDA Graphs |
|---|---:|---:|---:|---:|
| model initialization | 6.644 s | **6.052 s** | 6.516 s | 6.480 s |
| first forward | 0.542 s | 3.699 s | **0.156 s** | 3.372 s |
| median of 30 warm forwards | **9.69 ms** | 61.61 ms | 10.79 ms | 12.26 ms |
| process peak RAM | 4,583.1 MiB | 7,917.4 MiB | **1,048.3 MiB** | 1,219.2 MiB |
| runtime-reported GPU memory | 3,990.3 MiB | **3,844.4 MiB** | 3,877.1 MiB | 3,877.1 MiB |

Chomik won the warm forward: it was 1.11x faster than PyTorch eager, 1.26x
faster than CUDA Graphs, and 6.36x faster than tinygrad. The backend compiles
the graph once, flattens linear projections into 2D GEMMs, and uses one FP32
kernel for LayerNorm. The model contains 1,007,169,536 random FP32 parameters
(3.752 GiB for weights alone); the benchmark includes no embedding, tokenizer,
LM head, KV cache, or token generation.

### Training an approximately 1B-parameter LLM core

The training benchmark runs a synthetic MSE objective and SGD with `lr=1e-3`
on the same core. It measures the first step and the median of subsequent steps.
Chomik and PyTorch compatibility is checked through fingerprints of the
gradient and the updated final-normalization weight:

```bash
python benchmarks/llm_1b_training.py --steps 12
python benchmarks/llm_1b_training.py --steps 12 --inplace-sgd
python benchmarks/llm_1b_training.py --steps 12 --inplace-sgd --json
```

On an RTX 5070 Ti with FP32, `batch=1`, and sequence length 32, both frameworks
completed 12 steps without running out of memory. The run below uses
`--inplace-sgd` for Chomik:

| metric | Chomik CUDA | PyTorch eager |
|---|---:|---:|
| model initialization | 6.533 s | **6.485 s** |
| materializing Chomik weights on the GPU | 0.372 s | — |
| first `forward + backward + SGD` | **272.9 ms** | 293.6 ms |
| median of 11 warm steps | 65.4 ms | **51.9 ms** |
| process peak RAM | 4,588.1 MiB | **1,186.6 MiB** |
| runtime-reported GPU memory | 7,795.4 MiB | **7,750.4 MiB** |

The optimization removes reductions over length-one axes, routes singleton
batches to 2D GEMM, fuses softmax backward into one kernel, combines all three
LayerNorm gradients into one CUDA call, and performs the SGD update with one
kernel. Compared with the previous measurement, Chomik's first step fell from
335.8 ms to 272.9 ms, while its warm step fell from 91.1 ms to 65.4 ms, a 28.2%
improvement. Buffer lifetime analysis releases results after their final use
and reduced the default mode's peak from 11,817.8 MiB to 11,635.7 MiB. Optional
in-place SGD removes the new-weight copy and lowers the peak further to
7,795.4 MiB, only 45.0 MiB (0.6%) above PyTorch. Chomik's warm step remains
26.1% slower.

## Complete generation with a real 1.1B model

The `generate_tinyllama.py` example runs the real
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`: it downloads a pinned weight revision,
renders the chat template, tokenizes the prompt, runs the embedding, 22 Llama
blocks, and LM head, performs greedy decoding or sampling, and decodes the
response. Its 1,100,048,384 BF16 parameters remain in native MLX memory.

```bash
.venv/bin/python -m pip install '.[llm,mlx]'
.venv/bin/python examples/generate_tinyllama.py --compiler mlx
.venv/bin/python examples/generate_tinyllama.py \
  --prompt 'Explain lazy execution in one sentence.' \
  --temperature 0.7 --top-k 50 --max-new-tokens 32
```

The first run downloads about 2.2 GB into the standard Hugging Face cache; the
weights are not stored in the repository. The model revision is pinned to
`fe8a4ea1ffedaf415f4da2f062534de366a451e6` for reproducibility.

Prefill creates the KV cache, and every subsequent step updates it with a mask.
The embedding, RoPE, grouped-query attention, RMSNorm, SiLU, KV cache, and final
position selection are still composed exclusively from the six IR instructions
described above. The MLX backend can recognize portable RMSNorm, RoPE, and
attention subgraphs and lower them to faster kernels; other backends execute
their regular expansions. `TinyLlamaRuntime` materializes weights only once and
caches prefill programs by shape and decode programs by cache length. Token and
KV data are still bound separately for every request.

For the default prompt, the model generates:

```text
The capital of France is Paris.
```

The current MLX-LM comparison runs both runtimes in separate processes,
materializes weights before measurement, and checks token identity:

```bash
.venv/bin/python -m pip install '.[benchmark,llm]'
.venv/bin/python benchmarks/tinyllama_vs_mlx_lm.py --trials 9
```

Example result on an Apple M1 Max (`MLX 0.32.1`, `MLX-LM 0.31.3`, a 28-token
prompt, and eight response tokens):

| TinyLlama 1.1B BF16 | Chomik | MLX-LM |
|---|---:|---:|
| first token, cold graph | **0.077 s** | 0.121 s |
| complete response, cold graph | **0.157 s** | 0.201 s |
| first token after warm-up | **0.024 s** | 0.041 s |
| warm decode | 118.9 tokens/s | **130.4 tokens/s** |
| complete response after warm-up | **0.084 s** | 0.094 s |
| peak GPU memory | 2.118 GiB | 2.120 GiB |

Chomik reaches about 91% of native MLX-LM's decode throughput and has a shorter
TTFT for repeated shapes thanks to its whole-program cache. All eight token IDs
are identical in both implementations.

### TinyLlama experiment against tinygrad

The comparison was run on an Apple M1 Max, macOS 27.0, and Python 3.11.14.
Chomik used MLX 0.32.1. tinygrad came from the official
[`v0.14.0`](https://github.com/tinygrad/tinygrad/tree/v0.14.0) tag at commit
`6f87158`; the test used its `tinygrad.llm.model.Transformer` implementation,
`TinyJit`, and KV cache. The source tag was required because the 0.14.0 PyPI
wheel did not include the `tinygrad.llm.kernels` directory in the tested
environment.

Both frameworks received the same BF16 weights, activations, and KV cache, the
same pinned TinyLlama revision, the same 28-token chat prompt, `batch=1`, context
length 36, direct greedy argmax, and a limit of eight new tokens. Because the
official tinygrad model converts the embedding to FP32 and the cache to FP16,
those two casts were changed to BF16 in a temporary copy of the tag. Gumbel
sampling was replaced by direct `argmax`, which is equivalent at zero
temperature. The weights were already in the local cache, and loading time was
excluded. Each framework performed six generations. The warm result is the
median of the final four runs; the prompt cache was reset, so both frameworks
performed prefill again.

| TinyLlama 1.1B, Metal | Chomik | tinygrad 0.14.0 |
|---|---:|---:|
| first token, cold JIT | **0.078 s** | 2.585 s |
| complete eight tokens, cold JIT | **0.174 s** | 4.335 s |
| first token after capture | **0.024 s** | 0.389 s |
| complete eight tokens after capture | **0.085 s** | 0.550 s |
| warm decode | **117.9 tokens/s** | 43.6 tokens/s |
| device memory | 2.118 GiB | **2.052 GiB** |

In steady decode, Chomik was about 2.7x faster, while the complete short response
after warm-up took about 6.5x less time. The largest gap occurred with a cold
JIT: tinygrad needed two slow runs for capture and compilation. tinygrad used
about 3% less device memory.

Both implementations produced identical token IDs:

```text
1576, 7483, 310, 3444, 338, 3681, 29889, 2
The capital of France is Paris.
```

The comparison aligns stored tensor dtypes. Internal accumulation precision
remains a kernel-level choice in each framework. Generated token identity was
verified, not bitwise identity of every logit. The ratios may change with longer
contexts and responses.

### Why Chomik performs well against tinygrad

These results do not mean that Chomik is a more capable general-purpose
compiler than tinygrad. Chomik is deliberately narrower: its low-level IR has
only six operations, shapes are static during compilation, and a graph is
lowered once to a straight-line backend program. Structurally identical graphs
reuse the compiled program and replace only their dynamic inputs.

Chomik also delegates most device-specific optimization to mature native
runtimes. The MLX backend calls optimized MLX primitives, including fast
RMSNorm, RoPE, and scaled dot-product attention. The CUDA backend uses
CuPy/cuBLAS, flattens compatible projections into 2D GEMMs, and provides
specialized kernels for LayerNorm and selected backward passes. Parameters and
KV caches stay on the GPU, while synchronization is deferred until a result is
actually read.

tinygrad solves a broader problem. It performs more of its own scheduling,
kernel generation, buffer management, and device abstraction across a much
wider range of operations and hardware. That generality has a measurable
runtime and compilation cost, especially for small operations and short,
fixed-shape workloads. The reported tinygrad 0.14.0 runs used the default
`TinyJit` path without optional BEAM search or workload-specific tuning.

The comparison still controls the important numerical variables: frameworks
run in separate processes with the same weights, stored dtypes, inputs, and
shapes; warm-up is performed; transfers and result readback are included in the
microbenchmarks; and fingerprints, accuracy, or generated token IDs are checked.
The large gaps on small matrix multiplications therefore mostly measure runtime
overhead, while the gap becomes much smaller for large GEMMs and transformer
training.

There are two useful reality checks. Native MLX-LM remains faster in steady
TinyLlama decode (about 130 versus 118 tokens/s), and PyTorch eager wins all ten
general CUDA microbenchmarks and training cases. Chomik's strongest results come
from static graphs that match its backend lowerings particularly well. A new
Vulkan or other backend does not inherit this performance automatically; it
must provide equally good kernels and lowering choices while preserving the
same portable six-operation IR.

### Experimental speculative decoding

`LlamaDecoderBlock` verifies several positions in one graph using the same six
IR operations. The mechanism is not part of the MLX backend, so a future CUDA
or Vulkan backend can compile exactly the same block. The greedy runtime can
optionally use the pinned `Felladrin/Llama-68M-Chat-v1` as its draft model. At
load time it checks the complete 32,000-token mapping, and the target model
accepts candidates only up to the first mismatch:

```bash
PYTHONPATH=. .venv/bin/python examples/generate_tinyllama.py \
  --speculative-tokens 6 --temperature 0
```

The option is disabled by default. In BF16, the block target may use a different
Metal kernel than single-token decode and therefore change accumulation order.
This changes neither dtype nor the mathematical algorithm, but KV-cache
rounding after several accepted tokens can eventually lead to a different
`argmax`. The experiment therefore does not yet meet the strict identical-token
requirement.

The reproducible benchmark covers ten prompts, runs variants in separate
processes, and reports both elapsed time and full-sequence identity:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/tinyllama_speculative.py \
  --trials 3 --speculative-tokens 6
```

On an Apple M1 Max, eight of ten cases preserved identical tokens, while only
three became faster. Speedups in winning cases ranged from 1.00x to 1.06x; the
worst case was about 3.6x slower. The conclusion is that the portable
infrastructure works, but an independent 68M draft and the current block kernels
are not yet suitable as a default optimization. Safe enablement requires a
verification kernel with matching accumulation order and a draft trained for
the specific target.
