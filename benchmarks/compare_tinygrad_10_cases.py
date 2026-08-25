from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Dict, List, Sequence, Tuple

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
) -> Tuple[float, float]:
    from chomikgrad import Linear, ReLU, SGD, Sequential, Tensor, cross_entropy, no_grad

    rng = np.random.default_rng(7)
    model = Sequential(
        Linear(64, 48, rng=rng),
        ReLU(),
        Linear(48, 10, rng=rng),
    )
    optimizer = SGD(model.parameters(), lr=0.12)
    started = time.perf_counter()
    for _ in range(20):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), 64):
            indexes = order[start : start + 64]
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
    framework: str, trials: int, repeat_scale: float, device: str
) -> List[Dict[str, object]]:
    compiler = "mlx" if device == "metal" else "cuda"
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
        else:
            from tinygrad import Tensor as TinyTensor, TinyJit

            def raw(*inputs: object, name: str = name) -> object:
                return operation(name, inputs).realize()  # type: ignore[union-attr]

            compiled = TinyJit(raw)

            def run(arrays: Tuple[np.ndarray, ...] = arrays, compiled: object = compiled) -> np.ndarray:
                return compiled(*[TinyTensor(value) for value in arrays]).numpy()  # type: ignore[operator,union-attr]

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

    train_x, test_x, train_y, test_y = digits_data()
    mlp = (
        (lambda *values: chomik_mlp_trial(*values, compiler))
        if framework == "chomik"
        else tinygrad_mlp_trial
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
    )

    if framework == "chomik":
        transformer = lambda: benchmark_chomik(
            compiler, train_x, test_x, train_y, test_y, 7, 10, 64
        )
    else:
        transformer = lambda: benchmark_tinygrad(
            train_x, test_x, train_y, test_y, 7, 10, 64
        )
    results.append(training_result("train_transformer_10_epochs", transformer, trials))

    if framework == "chomik":
        from chomikgrad import get_compiler

        active_compiler = get_compiler(compiler)
        if hasattr(active_compiler, "close"):
            active_compiler.close()
    return results


def worker_command(framework: str, arguments: argparse.Namespace) -> List[Dict[str, object]]:
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
        "--trials",
        str(arguments.trials),
        "--repeat-scale",
        str(arguments.repeat_scale),
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
    raise RuntimeError(f"{framework} worker returned no benchmark result")


def compare(
    chomik: List[Dict[str, object]],
    tinygrad: List[Dict[str, object]],
    *,
    accuracy_atol: float = 1e-9,
) -> List[Dict[str, object]]:
    if len(chomik) != 10 or len(tinygrad) != 10:
        raise RuntimeError("expected exactly ten benchmark cases")
    compared = []
    for left, right in zip(chomik, tinygrad):
        if left["name"] != right["name"] or left["unit"] != right["unit"]:
            raise RuntimeError("benchmark workers returned incompatible cases")
        if "fingerprint" in left:
            np.testing.assert_allclose(
                left["fingerprint"],
                right["fingerprint"],
                rtol=2e-3,
                atol=2e-3,
            )
        if "accuracy" in left:
            np.testing.assert_allclose(
                left["accuracy"], right["accuracy"], rtol=0, atol=accuracy_atol
            )
        chomik_time = float(left["median"])
        tinygrad_time = float(right["median"])
        compared.append(
            {
                "name": left["name"],
                "unit": left["unit"],
                "chomik": chomik_time,
                "tinygrad": tinygrad_time,
                "winner": "chomik" if chomik_time < tinygrad_time else "tinygrad",
                "ratio": max(chomik_time, tinygrad_time)
                / min(chomik_time, tinygrad_time),
                "chomik_cold": left.get("cold"),
                "tinygrad_cold": right.get("cold"),
                "accuracy": left.get("accuracy"),
            }
        )
    return compared


def print_table(results: List[Dict[str, object]]) -> None:
    print("case,unit,chomik,tinygrad,winner,ratio")
    for result in results:
        print(
            f"{result['name']},{result['unit']},"
            f"{result['chomik']:.6f},{result['tinygrad']:.6f},"
            f"{result['winner']},{result['ratio']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare chomik-grad and tinygrad on ten GPU workloads"
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repeat-scale", type=float, default=1.0)
    parser.add_argument("--device", choices=("metal", "cuda"), default="metal")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--worker", choices=("chomik", "tinygrad"), help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.trials <= 0 or arguments.repeat_scale <= 0:
        parser.error("trials and repeat scale must be positive")

    if arguments.worker:
        print(
            RESULT_PREFIX
            + json.dumps(
                run_worker(
                    arguments.worker,
                    arguments.trials,
                    arguments.repeat_scale,
                    arguments.device,
                )
            )
        )
        return

    results = compare(
        worker_command("chomik", arguments),
        worker_command("tinygrad", arguments),
        accuracy_atol=0.01 if arguments.device == "cuda" else 1e-9,
    )
    if arguments.json:
        print(json.dumps(results, indent=2))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
