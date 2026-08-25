from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "CHOMIK_TINYLLAMA_RESULT="
PROMPT = "What is the capital of France? Answer in one short sentence."

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def chomik_runs(trials: int) -> List[Dict[str, object]]:
    from examples.generate_tinyllama import TinyLlamaRuntime

    arguments = SimpleNamespace(
        prompt=PROMPT,
        system=None,
        max_new_tokens=12,
        temperature=0.0,
        top_k=50,
        seed=0,
        model_dir=None,
        json=True,
    )
    runtime = TinyLlamaRuntime()
    return [runtime.generate(arguments) for _ in range(trials)]


def mlx_lm_runs(trials: int) -> List[Dict[str, object]]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from tokenizers import Tokenizer

    from examples.generate_tinyllama import (
        MODEL_REVISION,
        encode_chat,
        format_chat,
        model_directory,
    )

    directory = model_directory(None)
    config = json.loads((directory / "config.json").read_text())
    raw_tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    prompt_ids = encode_chat(
        raw_tokenizer,
        format_chat(PROMPT, None),
        int(config["eos_token_id"]),
    )
    model, _ = load(str(directory), lazy=False)
    mx.eval(model.parameters())
    sampler = make_sampler(temp=0.0)
    results = []
    for _ in range(trials):
        mx.reset_peak_memory()
        started = time.perf_counter()
        previous = started
        token_seconds = []
        token_ids = []
        for token, _ in generate_step(
            mx.array(prompt_ids),
            model,
            max_tokens=12,
            sampler=sampler,
        ):
            now = time.perf_counter()
            token_seconds.append(now - previous)
            previous = now
            token_ids.append(int(token))
            if token == int(config["eos_token_id"]):
                break
        elapsed = time.perf_counter() - started
        warm = token_seconds[1:]
        results.append(
            {
                "revision": MODEL_REVISION,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(token_ids),
                "generated_token_ids": token_ids,
                "first_token_seconds": token_seconds[0],
                "warm_tokens_per_second": 1 / statistics.median(warm),
                "total_generation_seconds": elapsed,
                "gpu_peak_mib": mx.get_peak_memory() / (1024 * 1024),
            }
        )
    return results


def worker(framework: str, trials: int) -> List[Dict[str, object]]:
    if framework == "chomik":
        return chomik_runs(trials)
    return mlx_lm_runs(trials)


def run_worker(framework: str, trials: int) -> List[Dict[str, object]]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        framework,
        "--trials",
        str(trials),
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


def summarize(runs: List[Dict[str, object]]) -> Dict[str, object]:
    warm = runs[1:] or runs

    def total(run: Dict[str, object]) -> float:
        return float(
            run.get("wall_generation_seconds", run["total_generation_seconds"])
        )

    return {
        "cold_first_token_seconds": runs[0]["first_token_seconds"],
        "cold_total_seconds": total(runs[0]),
        "warm_first_token_seconds": statistics.median(
            float(run["first_token_seconds"]) for run in warm
        ),
        "warm_decode_tokens_per_second": statistics.median(
            float(run["warm_tokens_per_second"]) for run in warm
        ),
        "warm_total_seconds": statistics.median(
            total(run) for run in warm
        ),
        "peak_gpu_mib": max(float(run["gpu_peak_mib"]) for run in runs),
        "generated_token_ids": runs[-1]["generated_token_ids"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare real TinyLlama generation in Chomik and MLX-LM"
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--worker", choices=("chomik", "mlx-lm"), help=argparse.SUPPRESS
    )
    arguments = parser.parse_args()
    if arguments.trials <= 0:
        parser.error("trials must be positive")
    if arguments.worker:
        print(
            RESULT_PREFIX
            + json.dumps(worker(arguments.worker, arguments.trials))
        )
        return

    chomik = summarize(run_worker("chomik", arguments.trials))
    mlx_lm = summarize(run_worker("mlx-lm", arguments.trials))
    if chomik["generated_token_ids"] != mlx_lm["generated_token_ids"]:
        raise RuntimeError("Chomik and MLX-LM generated different token IDs")
    result = {"chomik": chomik, "mlx_lm": mlx_lm}
    if arguments.json:
        print(json.dumps(result, indent=2))
        return
    print("metric,chomik,mlx-lm")
    for key in (
        "cold_first_token_seconds",
        "cold_total_seconds",
        "warm_first_token_seconds",
        "warm_decode_tokens_per_second",
        "warm_total_seconds",
        "peak_gpu_mib",
    ):
        print(f"{key},{chomik[key]:.6f},{mlx_lm[key]:.6f}")


if __name__ == "__main__":
    main()
