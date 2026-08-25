from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "CHOMIK_LLM_RESULT="
MIB = 1024 * 1024


def process_peak_mib() -> float:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        get_process_memory_info.restype = wintypes.BOOL
        process = get_current_process()
        if not get_process_memory_info(
            process, ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError()
        return float(counters.PeakWorkingSetSize / MIB)

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak / MIB if sys.platform == "darwin" else peak / 1024)


def fingerprint(value: np.ndarray) -> List[float]:
    data = np.asarray(value, dtype=np.float64)
    return [
        float(data.sum()),
        float(np.abs(data).sum()),
        float(np.square(data).sum()),
        float(data.min()),
        float(data.max()),
    ]


def expected_parameter_count(layers: int, width: int, hidden: int) -> int:
    attention = 4 * (width * width + width)
    feed_forward = hidden * width + hidden + width * hidden + width
    two_norms = 4 * width
    final_norm = 2 * width
    return layers * (attention + feed_forward + two_norms) + final_norm


class ChomikAdapter:
    name = "chomik"

    def __init__(self) -> None:
        from chomikgrad import Tensor, no_grad

        self.Tensor = Tensor
        self.inference_context = no_grad
        self.parameter_count = 0

    def parameter(self, value: np.ndarray) -> Any:
        self.parameter_count += value.size
        return self.Tensor(value)

    def constant(self, value: np.ndarray) -> Any:
        return self.Tensor(value)

    def input(self, value: np.ndarray) -> Any:
        return self.Tensor(value, copy=False)

    @staticmethod
    def mean(value: Any) -> Any:
        return value.mean(axis=-1, keepdims=True)

    @staticmethod
    def transpose_weight(value: Any) -> Any:
        return value.T


class TinygradAdapter:
    name = "tinygrad"

    def __init__(self) -> None:
        from tinygrad import Tensor

        self.Tensor = Tensor
        self.inference_context = nullcontext
        self.parameter_count = 0

    def parameter(self, value: np.ndarray) -> Any:
        self.parameter_count += value.size
        return self.Tensor(value)

    def constant(self, value: np.ndarray) -> Any:
        return self.Tensor(value)

    def input(self, value: np.ndarray) -> Any:
        return self.Tensor(value)

    @staticmethod
    def mean(value: Any) -> Any:
        return value.mean(axis=-1, keepdim=True)

    @staticmethod
    def transpose_weight(value: Any) -> Any:
        return value.transpose()


def random_weight(
    adapter: Any,
    rng: np.random.Generator,
    shape: Tuple[int, ...],
    scale: float,
) -> Any:
    value = rng.standard_normal(shape, dtype=np.float32)
    value *= np.float32(scale)
    return adapter.parameter(value)


class Linear:
    def __init__(
        self,
        adapter: Any,
        rng: np.random.Generator,
        inputs: int,
        outputs: int,
    ) -> None:
        self.adapter = adapter
        self.weight = random_weight(
            adapter,
            rng,
            (outputs, inputs),
            1.0 / np.sqrt(inputs),
        )
        self.bias = adapter.parameter(np.zeros(outputs, dtype=np.float32))

    def __call__(self, inputs: Any) -> Any:
        return inputs @ self.adapter.transpose_weight(self.weight) + self.bias


class LayerNorm:
    def __init__(self, adapter: Any, width: int) -> None:
        self.adapter = adapter
        self.weight = adapter.parameter(np.ones(width, dtype=np.float32))
        self.bias = adapter.parameter(np.zeros(width, dtype=np.float32))

    def __call__(self, inputs: Any) -> Any:
        mean = self.adapter.mean(inputs)
        centered = inputs - mean
        variance = self.adapter.mean(centered * centered)
        return centered / (variance + 1e-5).sqrt() * self.weight + self.bias


class CausalAttention:
    def __init__(
        self,
        adapter: Any,
        rng: np.random.Generator,
        width: int,
        heads: int,
        sequence: int,
    ) -> None:
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = Linear(adapter, rng, width, width)
        self.key = Linear(adapter, rng, width, width)
        self.value = Linear(adapter, rng, width, width)
        self.output = Linear(adapter, rng, width, width)
        mask = np.triu(
            np.full((sequence, sequence), -1e9, dtype=np.float32),
            k=1,
        ).reshape(1, 1, sequence, sequence)
        self.mask = adapter.constant(mask)

    def split_heads(self, inputs: Any) -> Any:
        batch, tokens, _ = inputs.shape
        return inputs.reshape(
            batch,
            tokens,
            self.heads,
            self.head_width,
        ).permute(0, 2, 1, 3)

    def __call__(self, inputs: Any) -> Any:
        query = self.split_heads(self.query(inputs))
        key = self.split_heads(self.key(inputs))
        value = self.split_heads(self.value(inputs))
        scores = (
            query @ key.transpose(-2, -1)
        ) / np.sqrt(self.head_width) + self.mask
        attended = scores.softmax(axis=-1) @ value
        batch, _, tokens, _ = attended.shape
        merged = attended.permute(0, 2, 1, 3).reshape(
            batch,
            tokens,
            self.width,
        )
        return self.output(merged)


class DecoderBlock:
    def __init__(
        self,
        adapter: Any,
        rng: np.random.Generator,
        width: int,
        heads: int,
        hidden: int,
        sequence: int,
    ) -> None:
        self.norm1 = LayerNorm(adapter, width)
        self.attention = CausalAttention(
            adapter,
            rng,
            width,
            heads,
            sequence,
        )
        self.norm2 = LayerNorm(adapter, width)
        self.linear1 = Linear(adapter, rng, width, hidden)
        self.linear2 = Linear(adapter, rng, hidden, width)

    def __call__(self, inputs: Any) -> Any:
        attended = inputs + self.attention(self.norm1(inputs))
        return attended + self.linear2(self.linear1(self.norm2(attended)).relu())


class DecoderCore:
    def __init__(
        self,
        adapter: Any,
        layers: int,
        width: int,
        heads: int,
        hidden: int,
        sequence: int,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.blocks = [
            DecoderBlock(adapter, rng, width, heads, hidden, sequence)
            for _ in range(layers)
        ]
        self.norm = LayerNorm(adapter, width)

    def __call__(self, inputs: Any) -> Any:
        output = inputs
        for block in self.blocks:
            output = block(output)
        return self.norm(output)


def gpu_peak_mib(framework: str, device: str) -> float:
    if framework == "chomik":
        if device == "metal":
            import mlx.core as mx

            return float(mx.get_peak_memory() / MIB)
        import cupy as cp

        return float(cp.get_default_memory_pool().total_bytes() / MIB)
    try:
        from tinygrad.engine.realize import GlobalCounters

        return float(GlobalCounters.mem_used / MIB)
    except (ImportError, AttributeError):
        return -1.0


def run_worker(arguments: argparse.Namespace) -> Dict[str, object]:
    if arguments.width % arguments.heads:
        raise ValueError("width must be divisible by heads")
    framework = arguments.worker
    compiler = "mlx" if arguments.device == "metal" else "cuda"
    adapter = ChomikAdapter() if framework == "chomik" else TinygradAdapter()

    if framework == "chomik":
        if arguments.device == "metal":
            import mlx.core as mx

            mx.reset_peak_memory()
        else:
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()

    started = time.perf_counter()
    model = DecoderCore(
        adapter,
        arguments.layers,
        arguments.width,
        arguments.heads,
        arguments.hidden,
        arguments.sequence,
        arguments.seed,
    )
    initialization_seconds = time.perf_counter() - started
    expected = expected_parameter_count(
        arguments.layers,
        arguments.width,
        arguments.hidden,
    )
    if adapter.parameter_count != expected:
        raise RuntimeError(
            f"expected {expected} parameters, built {adapter.parameter_count}"
        )

    input_rng = np.random.default_rng(arguments.seed + 1)
    input_value = input_rng.standard_normal(
        (arguments.batch, arguments.sequence, arguments.width),
        dtype=np.float32,
    )

    if framework == "tinygrad":
        from tinygrad import TinyJit

        def raw_forward(inputs: Any) -> Any:
            return model(inputs).realize()

        compiled = TinyJit(raw_forward)

        def forward() -> np.ndarray:
            return compiled(adapter.input(input_value)).numpy()
    else:
        def forward() -> np.ndarray:
            with adapter.inference_context():
                return model(adapter.input(input_value)).numpy(compiler=compiler)

    early_times = []
    result = None
    for _ in range(3):
        call_started = time.perf_counter()
        result = forward()
        early_times.append(time.perf_counter() - call_started)

    warm_times = []
    for _ in range(arguments.warm_runs):
        call_started = time.perf_counter()
        result = forward()
        warm_times.append(time.perf_counter() - call_started)

    assert result is not None
    output_fingerprint = fingerprint(result)
    result_data = {
        "framework": framework,
        "parameters": adapter.parameter_count,
        "parameter_gib_fp32": adapter.parameter_count * 4 / (1024**3),
        "initialization_seconds": initialization_seconds,
        "first_seconds": early_times[0],
        "capture_seconds": early_times[1],
        "warm_median_seconds": float(np.median(warm_times)),
        "warm_runs": arguments.warm_runs,
        "process_peak_mib": process_peak_mib(),
        "gpu_peak_mib": gpu_peak_mib(framework, arguments.device),
        "fingerprint": output_fingerprint,
    }

    if framework == "chomik":
        from chomikgrad import get_compiler

        active_compiler = get_compiler(compiler)
        if hasattr(active_compiler, "close"):
            active_compiler.close()
    return result_data


def run_subprocess(framework: str, arguments: argparse.Namespace) -> Dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT)
    if framework == "tinygrad":
        environment["DEV"] = arguments.device.upper()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        framework,
        "--layers",
        str(arguments.layers),
        "--width",
        str(arguments.width),
        "--heads",
        str(arguments.heads),
        "--hidden",
        str(arguments.hidden),
        "--sequence",
        str(arguments.sequence),
        "--batch",
        str(arguments.batch),
        "--warm-runs",
        str(arguments.warm_runs),
        "--seed",
        str(arguments.seed),
        "--device",
        arguments.device,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{framework} worker failed with {completed.returncode}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    raise RuntimeError(f"{framework} worker returned no result")


def compare(arguments: argparse.Namespace) -> Dict[str, object]:
    chomik = run_subprocess("chomik", arguments)
    tinygrad = run_subprocess("tinygrad", arguments)
    if chomik["parameters"] != tinygrad["parameters"]:
        raise RuntimeError("frameworks built different parameter counts")
    np.testing.assert_allclose(
        chomik["fingerprint"],
        tinygrad["fingerprint"],
        rtol=5e-2,
        atol=5e-2,
    )
    return {
        "configuration": {
            "layers": arguments.layers,
            "width": arguments.width,
            "heads": arguments.heads,
            "hidden": arguments.hidden,
            "sequence": arguments.sequence,
            "batch": arguments.batch,
        },
        "chomik": chomik,
        "tinygrad": tinygrad,
        "warm_speed_ratio": float(tinygrad["warm_median_seconds"])
        / float(chomik["warm_median_seconds"]),
    }


def print_result(result: Dict[str, object]) -> None:
    configuration = result["configuration"]
    chomik = result["chomik"]
    tinygrad = result["tinygrad"]
    print(
        "configuration "
        f"layers={configuration['layers']} width={configuration['width']} "
        f"heads={configuration['heads']} hidden={configuration['hidden']} "
        f"sequence={configuration['sequence']} batch={configuration['batch']}"
    )
    print(
        f"parameters={chomik['parameters']} "
        f"fp32_parameter_gib={chomik['parameter_gib_fp32']:.3f}"
    )
    print("framework,init_s,first_s,capture_s,warm_s,process_peak_mib,gpu_peak_mib")
    for framework in (chomik, tinygrad):
        print(
            f"{framework['framework']},{framework['initialization_seconds']:.6f},"
            f"{framework['first_seconds']:.6f},{framework['capture_seconds']:.6f},"
            f"{framework['warm_median_seconds']:.6f},"
            f"{framework['process_peak_mib']:.1f},{framework['gpu_peak_mib']:.1f}"
        )
    print(f"tinygrad_over_chomik_warm={result['warm_speed_ratio']:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a roughly 1B-parameter decoder core on GPU"
    )
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--sequence", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warm-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("metal", "cuda"), default="metal")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--worker", choices=("chomik", "tinygrad"), help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    positive = (
        arguments.layers,
        arguments.width,
        arguments.heads,
        arguments.hidden,
        arguments.sequence,
        arguments.batch,
        arguments.warm_runs,
    )
    if any(value <= 0 for value in positive):
        parser.error("model dimensions and warm runs must be positive")

    if arguments.worker:
        print(RESULT_PREFIX + json.dumps(run_worker(arguments)))
        return

    result = compare(arguments)
    if arguments.json:
        print(json.dumps(result, indent=2))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
