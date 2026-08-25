from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "CHOMIK_BENCHMARK_RESULT="
MICRO_REPEATS = {
    "elementwise_1m": 30,
    "reduce_sum_4m": 30,
    "softmax_1024x1024": 30,
    "matmul_64": 150,
    "matmul_256": 100,
    "matmul_1024": 25,
    "matmul_2048": 10,
    "batched_matmul_16x4x64": 50,
}


def micro_cases() -> List[Tuple[str, Tuple[np.ndarray, ...]]]:
    rng = np.random.default_rng(20260825)
    cases = []
    cases.append(
        (
            "elementwise_1m",
            (
                rng.normal(size=(1_048_576,)).astype(np.float32),
                rng.normal(size=(1_048_576,)).astype(np.float32),
            ),
        )
    )
    cases.append(
        (
            "reduce_sum_4m",
            (rng.normal(size=(4_194_304,)).astype(np.float32),),
        )
    )
    cases.append(
        (
            "softmax_1024x1024",
            (rng.normal(size=(1024, 1024)).astype(np.float32),),
        )
    )
    for size in (64, 256, 1024, 2048):
        cases.append(
            (
                f"matmul_{size}",
                (
                    rng.normal(size=(size, size)).astype(np.float32),
                    rng.normal(size=(size, size)).astype(np.float32),
                ),
            )
        )
    cases.append(
        (
            "batched_matmul_16x4x64",
            (
                rng.normal(size=(16, 4, 64, 64)).astype(np.float32),
                rng.normal(size=(16, 4, 64, 64)).astype(np.float32),
            ),
        )
    )
    return cases


def operation(name: str, inputs: Sequence[object]) -> object:
    if name == "elementwise_1m":
        left, right = inputs
        return ((left * 1.1 + right).relu() * 0.5 - left / 3.0)  # type: ignore[operator,union-attr]
    if name == "reduce_sum_4m":
        return inputs[0].sum()  # type: ignore[union-attr]
    if name == "softmax_1024x1024":
        return inputs[0].softmax(axis=-1)  # type: ignore[union-attr]
    return inputs[0] @ inputs[1]  # type: ignore[operator]


def timed_median(
    function: Callable[[], np.ndarray], repeats: int
) -> Tuple[float, List[float]]:
    result = function()
    for _ in range(4):
        result = function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = function()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    value = np.asarray(result, dtype=np.float64)
    fingerprint = [
        float(value.sum()),
        float(np.abs(value).sum()),
        float(np.square(value).sum()),
        float(value.min()),
        float(value.max()),
    ]
    return float(np.median(samples)), fingerprint


def digits_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    digits = load_digits()
    features = (digits.data / 16.0).astype(np.float32)
    labels = digits.target.astype(np.int64)
    return train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=7,
        stratify=labels,
    )


def chomik_mlp_trial(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    compiler: str,
    *,
    compiled: bool = False,
) -> Tuple[float, float]:
    from chomikgrad import (
        Linear,
        ReLU,
        SGD,
        Sequential,
        Tensor,
        compile_train_step,
        cross_entropy,
        no_grad,
    )

    rng = np.random.default_rng(7)
    model = Sequential(
        Linear(64, 48, rng=rng),
        ReLU(),
        Linear(48, 10, rng=rng),
    )
    optimizer = SGD(model.parameters(), lr=0.12)
    started = time.perf_counter()
    steps = {}
    if compiled:

        def loss_function(inputs: Tensor, targets: Tensor) -> Tensor:
            logits = model(inputs)
            return (
                -(logits.log_softmax(axis=1) * targets).sum()
                / inputs.shape[0]
            )

        for size in {64, len(train_x) % 64} - {0}:
            steps[size] = compile_train_step(
                loss_function,
                optimizer,
                Tensor.zeros((size, 64)),
                Tensor.zeros((size, 10)),
                compiler=compiler,
            )
    for _ in range(20):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), 64):
            indexes = order[start : start + 64]
            if compiled:
                one_hot = np.zeros((len(indexes), 10), dtype=np.float32)
                one_hot[np.arange(len(indexes)), train_y[indexes]] = 1
                steps[len(indexes)](train_x[indexes], one_hot)
            else:
                optimizer.zero_grad()
                loss = cross_entropy(
                    model(Tensor(train_x[indexes], copy=False)),
                    train_y[indexes],
                )
                loss.backward()
                optimizer.step(compiler=compiler)
    model.parameters()[0].numpy(compiler=compiler)
    elapsed = time.perf_counter() - started
    with no_grad():
        predictions = (
            model(Tensor(test_x, copy=False))
            .numpy(compiler=compiler)
            .argmax(axis=1)
        )
    return elapsed, float((predictions == test_y).mean())


def tinygrad_mlp_trial(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
) -> Tuple[float, float]:
    from tinygrad import Tensor as TinyTensor, TinyJit
    from tinygrad.helpers import Context
    from tinygrad.nn.optim import SGD as TinySGD

    from chomikgrad import Linear, ReLU, Sequential

    rng = np.random.default_rng(7)
    reference = Sequential(
        Linear(64, 48, rng=rng),
        ReLU(),
        Linear(48, 10, rng=rng),
    )
    weight1, bias1, weight2, bias2 = [
        TinyTensor(parameter.numpy().copy()) for parameter in reference.parameters()
    ]
    parameters = [weight1, bias1, weight2, bias2]
    optimizer = TinySGD(parameters, lr=0.12)

    def model(inputs: object) -> object:
        hidden = (inputs @ weight1.transpose() + bias1).relu()  # type: ignore[operator,union-attr]
        return hidden @ weight2.transpose() + bias2  # type: ignore[operator,no-any-return]

    def raw_step(inputs: object, targets: object) -> object:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = -(logits.log_softmax(axis=1) * targets).sum() / inputs.shape[0]  # type: ignore[operator,union-attr]
        loss.backward()
        optimizer.step()
        return loss

    step = TinyJit(raw_step)
    tail_step = TinyJit(raw_step)
    started = time.perf_counter()
    with Context(TRAINING=1):
        for _ in range(20):
            order = rng.permutation(len(train_x))
            for start in range(0, len(order), 64):
                indexes = order[start : start + 64]
                one_hot = np.zeros((len(indexes), 10), dtype=np.float32)
                one_hot[np.arange(len(indexes)), train_y[indexes]] = 1
                selected = step if len(indexes) == 64 else tail_step
                selected(TinyTensor(train_x[indexes]), TinyTensor(one_hot))
    parameters[0].numpy()
    elapsed = time.perf_counter() - started
    predictions = model(TinyTensor(test_x)).argmax(axis=1).numpy()  # type: ignore[union-attr]
    return elapsed, float((predictions == test_y).mean())


def torch_mlp_trial(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    *,
    compiled: bool,
    compile_backend: str,
) -> Tuple[float, float]:
    try:
        import torch
    except ImportError as error:
        raise SystemExit("install PyTorch CUDA to run this benchmark") from error
    if not torch.cuda.is_available():
        raise RuntimeError("the PyTorch benchmark requires an available CUDA GPU")

    from chomikgrad import Linear, ReLU, Sequential
    from benchmarks.transformer_vs_tinygrad import compile_torch

    rng = np.random.default_rng(7)
    reference = Sequential(
        Linear(64, 48, rng=rng),
        ReLU(),
        Linear(48, 10, rng=rng),
    )
    arrays = [parameter.numpy().copy() for parameter in reference.parameters()]
    weight1 = torch.from_numpy(arrays[0]).cuda().requires_grad_()
    bias1 = torch.from_numpy(arrays[1]).cuda().requires_grad_()
    weight2 = torch.from_numpy(arrays[2]).cuda().requires_grad_()
    bias2 = torch.from_numpy(arrays[3]).cuda().requires_grad_()
    parameters = [weight1, bias1, weight2, bias2]
    optimizer = torch.optim.SGD(parameters, lr=0.12)

    def raw_forward(inputs: object) -> object:
        hidden = (inputs @ weight1.T + bias1).relu()  # type: ignore[operator,union-attr]
        return hidden @ weight2.T + bias2  # type: ignore[operator,no-any-return]

    def raw_tail_forward(inputs: object) -> object:
        return raw_forward(inputs)

    if compiled:
        forward = compile_torch(raw_forward, compile_backend)
        tail_forward = compile_torch(raw_tail_forward, compile_backend)
    else:
        forward = raw_forward
        tail_forward = raw_forward
    started = time.perf_counter()
    for _ in range(20):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), 64):
            indexes = order[start : start + 64]
            one_hot = np.zeros((len(indexes), 10), dtype=np.float32)
            one_hot[np.arange(len(indexes)), train_y[indexes]] = 1
            inputs = torch.from_numpy(train_x[indexes]).cuda()
            targets = torch.from_numpy(one_hot).cuda()
            optimizer.zero_grad(set_to_none=True)
            selected = forward if len(indexes) == 64 else tail_forward
            logits = selected(inputs)
            loss = -(logits.log_softmax(dim=1) * targets).sum() / inputs.shape[0]  # type: ignore[union-attr]
            loss.backward()  # type: ignore[union-attr]
            optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        predictions = (
            raw_forward(torch.from_numpy(test_x).cuda())
            .argmax(dim=1)  # type: ignore[union-attr]
            .cpu()
            .numpy()
        )
    return elapsed, float((predictions == test_y).mean())


def training_result(
    name: str,
    trial: Callable[[], Tuple[float, float]],
    trials: int,
) -> Dict[str, object]:
    durations = []
    accuracies = []
    for _ in range(trials):
        duration, accuracy = trial()
        durations.append(duration)
        accuracies.append(accuracy)
    warm = durations[1:] if len(durations) > 1 else durations
    return {
        "name": name,
        "unit": "s",
        "median": float(np.median(warm)),
        "cold": durations[0],
        "accuracy": float(np.median(accuracies)),
        "trials": trials,
    }


def run_worker(
    framework: str,
    trials: int,
    repeat_scale: float,
    device: str,
    torch_compile_backend: str,
    micro_only: bool = False,
    chomik_jit: bool = False,
) -> List[Dict[str, object]]:
    compiler = {
        "metal": "mlx",
        "cuda": "cuda",
        "opencl": "opencl",
        "vulkan": "vulkan",
    }[device]
    results = []
    for name, arrays in micro_cases():
        repeats = max(1, round(MICRO_REPEATS[name] * repeat_scale))
        if framework == "chomik":
            from chomikgrad import Tensor

            def run(arrays: Tuple[np.ndarray, ...] = arrays, name: str = name) -> np.ndarray:
                return operation(
                    name,
                    [Tensor(value, copy=False) for value in arrays],
                ).numpy(compiler=compiler)  # type: ignore[union-attr]
        elif framework == "tinygrad":
            from tinygrad import Tensor as TinyTensor, TinyJit

            def raw(*inputs: object, name: str = name) -> object:
                return operation(name, inputs).realize()  # type: ignore[union-attr]

            compiled = TinyJit(raw)

            def run(arrays: Tuple[np.ndarray, ...] = arrays, compiled: object = compiled) -> np.ndarray:
                return compiled(*[TinyTensor(value) for value in arrays]).numpy()  # type: ignore[operator,union-attr]
        else:
            import torch

            from benchmarks.transformer_vs_tinygrad import compile_torch

            def raw(*inputs: object, name: str = name) -> object:
                return operation(name, inputs)

            selected = (
                compile_torch(raw, torch_compile_backend)
                if framework == "torch-compile"
                else raw
            )

            def run(
                arrays: Tuple[np.ndarray, ...] = arrays,
                selected: object = selected,
            ) -> np.ndarray:
                output = selected(  # type: ignore[operator]
                    *[torch.from_numpy(value).cuda() for value in arrays]
                )
                return output.detach().cpu().numpy()  # type: ignore[union-attr]

        median, fingerprint = timed_median(run, repeats)
        results.append(
            {
                "name": name,
                "unit": "ms",
                "median": median,
                "fingerprint": fingerprint,
                "repeats": repeats,
            }
        )

    if micro_only:
        if framework == "chomik":
            from chomikgrad import get_compiler

            active_compiler = get_compiler(compiler)
            if hasattr(active_compiler, "close"):
                active_compiler.close()
        return results

    train_x, test_x, train_y, test_y = digits_data()
    if framework == "chomik":
        mlp = lambda *values: chomik_mlp_trial(
            *values, compiler, compiled=chomik_jit
        )
    elif framework == "tinygrad":
        mlp = tinygrad_mlp_trial
    else:
        mlp = lambda *values: torch_mlp_trial(
            *values,
            compiled=framework == "torch-compile",
            compile_backend=torch_compile_backend,
        )
    results.append(
        training_result(
            "train_mlp_20_epochs",
            lambda: mlp(train_x, test_x, train_y, test_y),
            trials,
        )
    )

    from benchmarks.transformer_vs_tinygrad import (
        benchmark_chomik,
        benchmark_tinygrad,
        benchmark_torch,
    )

    if framework == "chomik":
        transformer = lambda: benchmark_chomik(
            compiler,
            train_x,
            test_x,
            train_y,
            test_y,
            7,
            10,
            64,
            compiled=chomik_jit,
        )
    elif framework == "tinygrad":
        transformer = lambda: benchmark_tinygrad(
            train_x, test_x, train_y, test_y, 7, 10, 64
        )
    else:
        transformer = lambda: benchmark_torch(
            train_x,
            test_x,
            train_y,
            test_y,
            7,
            10,
            64,
            compiled=framework == "torch-compile",
            compile_backend=torch_compile_backend,
        )
    results.append(training_result("train_transformer_10_epochs", transformer, trials))

    if framework == "chomik":
        from chomikgrad import get_compiler

        active_compiler = get_compiler(compiler)
        if hasattr(active_compiler, "close"):
            active_compiler.close()
    elif framework.startswith("torch-"):
        import torch

        torch.cuda.synchronize()
    return results


def worker_command(framework: str, arguments: argparse.Namespace) -> List[Dict[str, object]]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT)
    if framework == "tinygrad":
        environment["DEV"] = {
            "opencl": "CL",
            "vulkan": "WEBGPU",
        }.get(arguments.device, arguments.device.upper())
        if arguments.device == "vulkan":
            environment["WGPU_BACKEND_TYPE"] = "Vulkan"
            environment["WEBGPU_BACKEND"] = "WGPUBackendType_Vulkan"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        framework,
        "--trials",
        str(arguments.trials),
        "--repeat-scale",
        str(arguments.repeat_scale),
        "--device",
        arguments.device,
        "--torch-compile-backend",
        arguments.torch_compile_backend,
    ]
    if arguments.micro_only:
        command.append("--micro-only")
    if arguments.chomik_jit:
        command.append("--chomik-jit")
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
    raise RuntimeError(f"{framework} worker returned no benchmark result")


def compare(
    framework_results: Mapping[str, List[Dict[str, object]]],
    *,
    accuracy_atol: float = 1e-9,
) -> List[Dict[str, object]]:
    if not framework_results:
        raise RuntimeError("expected benchmark results")
    expected = len(next(iter(framework_results.values())))
    if expected not in (8, 10) or any(
        len(results) != expected for results in framework_results.values()
    ):
        raise RuntimeError("expected eight or ten compatible benchmark cases")
    frameworks = tuple(framework_results)
    reference = framework_results[frameworks[0]]
    compared = []
    for index, left in enumerate(reference):
        cases = {
            framework: framework_results[framework][index]
            for framework in frameworks
        }
        for framework, candidate in cases.items():
            if (
                left["name"] != candidate["name"]
                or left["unit"] != candidate["unit"]
            ):
                raise RuntimeError("benchmark workers returned incompatible cases")
            if framework != frameworks[0] and "fingerprint" in left:
                np.testing.assert_allclose(
                    left["fingerprint"],
                    candidate["fingerprint"],
                    rtol=2e-3,
                    atol=2e-3,
                )
            if framework != frameworks[0] and "accuracy" in left:
                np.testing.assert_allclose(
                    left["accuracy"],
                    candidate["accuracy"],
                    rtol=0,
                    atol=accuracy_atol,
                )
        times = {
            framework: float(candidate["median"])
            for framework, candidate in cases.items()
        }
        winner = min(times, key=times.__getitem__)
        result: Dict[str, object] = {
            "name": left["name"],
            "unit": left["unit"],
            **times,
            "winner": winner,
            "ratio": max(times.values()) / min(times.values()),
            "cold": {
                framework: candidate.get("cold")
                for framework, candidate in cases.items()
            },
            "accuracy": {
                framework: candidate.get("accuracy")
                for framework, candidate in cases.items()
                if "accuracy" in candidate
            },
        }
        compared.append(result)
    return compared


def print_table(results: List[Dict[str, object]]) -> None:
    frameworks = tuple(
        key
        for key in ("chomik", "tinygrad", "torch-eager", "torch-compile")
        if key in results[0]
    )
    print("case,unit," + ",".join(frameworks) + ",winner,ratio")
    for result in results:
        print(
            f"{result['name']},{result['unit']},"
            + ",".join(f"{float(result[name]):.6f}" for name in frameworks)
            + ","
            f"{result['winner']},{result['ratio']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare chomik-grad, tinygrad, and PyTorch on ten GPU workloads"
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repeat-scale", type=float, default=1.0)
    parser.add_argument(
        "--device",
        choices=("metal", "cuda", "opencl", "vulkan"),
        default="metal",
    )
    parser.add_argument(
        "--torch-compile-backend",
        choices=("cudagraphs", "inductor"),
        default="cudagraphs",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--micro-only",
        action="store_true",
        help="run only the eight tensor microbenchmarks",
    )
    parser.add_argument(
        "--chomik-jit",
        action="store_true",
        help="capture and reuse Chomik training graphs",
    )
    parser.add_argument(
        "--worker",
        choices=("chomik", "tinygrad", "torch-eager", "torch-compile"),
        help=argparse.SUPPRESS,
    )
    arguments = parser.parse_args()
    if arguments.trials <= 0 or arguments.repeat_scale <= 0:
        parser.error("trials and repeat scale must be positive")
    if arguments.chomik_jit and arguments.device != "opencl":
        parser.error("--chomik-jit currently requires --device opencl")

    if arguments.worker:
        print(
            RESULT_PREFIX
            + json.dumps(
                run_worker(
                    arguments.worker,
                    arguments.trials,
                    arguments.repeat_scale,
                    arguments.device,
                    arguments.torch_compile_backend,
                    arguments.micro_only,
                    arguments.chomik_jit,
                )
            )
        )
        return

    frameworks = ["chomik", "tinygrad"]
    if arguments.device == "cuda":
        frameworks.extend(("torch-eager", "torch-compile"))
    results = compare(
        {
            framework: worker_command(framework, arguments)
            for framework in frameworks
        },
        accuracy_atol=0.01 if arguments.device != "metal" else 1e-9,
    )
    if arguments.json:
        print(json.dumps(results, indent=2))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
