from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.llm_1b_inference import (
    DecoderCore,
    expected_parameter_count,
    fingerprint,
    process_peak_mib,
)


RESULT_PREFIX = "CHOMIK_TRAINING_RESULT="
MIB = 1024 * 1024


class ChomikTrainingAdapter:
    def __init__(self) -> None:
        from chomikgrad import Parameter, SGD, Tensor, get_compiler

        self.Parameter = Parameter
        self.SGD = SGD
        self.Tensor = Tensor
        self.compiler = get_compiler("cuda")
        self.parameters: List[Any] = []
        self.parameter_count = 0

    def parameter(self, value: np.ndarray) -> Any:
        parameter = self.Parameter(value)
        self.parameters.append(parameter)
        self.parameter_count += value.size
        return parameter

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

    @staticmethod
    def layer_norm(
        inputs: Any,
        weight: Any,
        bias: Any,
        epsilon: float,
    ) -> Any:
        return inputs.layer_norm(weight, bias, epsilon)

    def materialize_parameters(self) -> None:
        for parameter in self.parameters:
            self.compiler._load_input(parameter._node)
        self.compiler.device.synchronize()


class TorchTrainingAdapter:
    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.parameters: List[Any] = []
        self.parameter_count = 0

    def parameter(self, value: np.ndarray) -> Any:
        parameter = self.torch.from_numpy(value).cuda().requires_grad_()
        self.parameters.append(parameter)
        self.parameter_count += value.size
        return parameter

    def constant(self, value: np.ndarray) -> Any:
        return self.torch.from_numpy(value).cuda()

    def input(self, value: np.ndarray) -> Any:
        return self.torch.from_numpy(value).cuda()

    @staticmethod
    def mean(value: Any) -> Any:
        return value.mean(dim=-1, keepdim=True)

    @staticmethod
    def transpose_weight(value: Any) -> Any:
        return value.T

    @staticmethod
    def layer_norm(
        inputs: Any,
        weight: Any,
        bias: Any,
        epsilon: float,
    ) -> Any:
        mean = inputs.mean(dim=-1, keepdim=True)
        centered = inputs - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        return centered / (variance + epsilon).sqrt() * weight + bias


def run_worker(arguments: argparse.Namespace) -> Dict[str, object]:
    framework = arguments.worker
    if framework == "chomik":
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        adapter: Any = ChomikTrainingAdapter()
    else:
        import torch

        torch.cuda.empty_cache()
        adapter = TorchTrainingAdapter()

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
    if framework == "torch":
        adapter.torch.cuda.synchronize()
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

    materialization_seconds = 0.0
    if framework == "chomik":
        started = time.perf_counter()
        adapter.materialize_parameters()
        materialization_seconds = time.perf_counter() - started
    else:
        adapter.torch.cuda.reset_peak_memory_stats()

    rng = np.random.default_rng(arguments.seed + 1)
    shape = (arguments.batch, arguments.sequence, arguments.width)
    inputs = adapter.input(rng.standard_normal(shape, dtype=np.float32))
    target = adapter.input(rng.standard_normal(shape, dtype=np.float32))
    optimizer = (
        adapter.SGD(
            adapter.parameters,
            lr=arguments.learning_rate,
            inplace=arguments.inplace_sgd,
        )
        if framework == "chomik"
        else adapter.torch.optim.SGD(
            adapter.parameters, lr=arguments.learning_rate
        )
    )

    step_times = []
    for _ in range(arguments.steps):
        optimizer.zero_grad()
        gc.collect()
        started = time.perf_counter()
        output = model(inputs)
        difference = output - target
        loss = (difference * difference).mean()
        loss.backward()
        if framework == "chomik":
            optimizer.step(compiler="cuda")
            adapter.compiler.device.synchronize()
        else:
            optimizer.step()
            adapter.torch.cuda.synchronize()
        step_times.append(time.perf_counter() - started)
        del output, difference, loss

    selected = adapter.parameters[-2]
    if framework == "chomik":
        gradient = selected.grad.numpy(compiler="cuda")
        updated = selected.numpy(compiler="cuda")
        gpu_peak_mib = cp.get_default_memory_pool().total_bytes() / MIB
    else:
        gradient = selected.grad.detach().cpu().numpy()
        updated = selected.detach().cpu().numpy()
        gpu_peak_mib = adapter.torch.cuda.max_memory_allocated() / MIB

    return {
        "framework": "chomik" if framework == "chomik" else "torch-eager",
        "parameters": adapter.parameter_count,
        "initialization_seconds": initialization_seconds,
        "materialization_seconds": materialization_seconds,
        "first_step_seconds": step_times[0],
        "warm_step_median_seconds": float(np.median(step_times[1:])),
        "warm_steps": step_times[1:],
        "gradient_fingerprint": fingerprint(gradient),
        "updated_parameter_fingerprint": fingerprint(updated),
        "process_peak_mib": process_peak_mib(),
        "gpu_peak_mib": gpu_peak_mib,
    }


def run_subprocess(framework: str, arguments: argparse.Namespace) -> Dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
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
        "--steps",
        str(arguments.steps),
        "--learning-rate",
        str(arguments.learning_rate),
        "--seed",
        str(arguments.seed),
    ]
    if arguments.inplace_sgd:
        command.append("--inplace-sgd")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(f"worker {framework} did not produce a result")


def compare(arguments: argparse.Namespace) -> Dict[str, object]:
    chomik = run_subprocess("chomik", arguments)
    torch = run_subprocess("torch", arguments)
    if chomik["parameters"] != torch["parameters"]:
        raise RuntimeError("frameworks built different parameter counts")
    np.testing.assert_allclose(
        chomik["gradient_fingerprint"],
        torch["gradient_fingerprint"],
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        chomik["updated_parameter_fingerprint"],
        torch["updated_parameter_fingerprint"],
        rtol=1e-5,
        atol=1e-6,
    )
    return {
        "configuration": {
            "layers": arguments.layers,
            "width": arguments.width,
            "heads": arguments.heads,
            "hidden": arguments.hidden,
            "sequence": arguments.sequence,
            "batch": arguments.batch,
            "steps": arguments.steps,
            "learning_rate": arguments.learning_rate,
            "inplace_sgd": arguments.inplace_sgd,
        },
        "frameworks": {"chomik": chomik, "torch-eager": torch},
        "winner": min(
            ("chomik", "torch-eager"),
            key=lambda name: float(
                {"chomik": chomik, "torch-eager": torch}[name][
                    "warm_step_median_seconds"
                ]
            ),
        ),
    }


def print_result(result: Dict[str, object]) -> None:
    configuration = result["configuration"]
    frameworks = result["frameworks"]
    print(
        "configuration "
        f"layers={configuration['layers']} width={configuration['width']} "
        f"heads={configuration['heads']} hidden={configuration['hidden']} "
        f"sequence={configuration['sequence']} batch={configuration['batch']} "
        f"steps={configuration['steps']} "
        f"inplace_sgd={configuration['inplace_sgd']}"
    )
    print("framework,init_s,materialize_s,first_step_s,warm_step_s,ram_mib,gpu_mib")
    for framework in frameworks.values():
        print(
            f"{framework['framework']},{framework['initialization_seconds']:.6f},"
            f"{framework['materialization_seconds']:.6f},"
            f"{framework['first_step_seconds']:.6f},"
            f"{framework['warm_step_median_seconds']:.6f},"
            f"{framework['process_peak_mib']:.1f},{framework['gpu_peak_mib']:.1f}"
        )
    print(f"warm_winner={result['winner']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one-billion-parameter training on CUDA"
    )
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--sequence", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--inplace-sgd",
        action="store_true",
        help="update Chomik CUDA parameters in place to reduce peak memory",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--worker", choices=("chomik", "torch"), help=argparse.SUPPRESS
    )
    arguments = parser.parse_args()
    if arguments.width % arguments.heads:
        parser.error("width must be divisible by heads")
    if min(
        arguments.layers,
        arguments.width,
        arguments.heads,
        arguments.hidden,
        arguments.sequence,
        arguments.batch,
    ) <= 0:
        parser.error("model dimensions must be positive")
    if arguments.steps < 2:
        parser.error("steps must be at least 2")
    if arguments.learning_rate <= 0:
        parser.error("learning rate must be positive")

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
