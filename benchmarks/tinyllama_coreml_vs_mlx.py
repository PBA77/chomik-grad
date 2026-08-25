from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from types import SimpleNamespace
from typing import Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "CHOMIK_COREML_RESULT="
PROMPT = "What is the capital of France? Answer in one short sentence."

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def worker(compiler: str, trials: int) -> Dict[str, object]:
    from examples.generate_tinyllama import TinyLlamaRuntime

    arguments = SimpleNamespace(
        prompt=PROMPT,
        system=None,
        max_new_tokens=8,
        temperature=0.0,
        top_k=50,
        seed=0,
        model_dir=None,
        json=True,
    )
    runtime = TinyLlamaRuntime(compiler=compiler, dtype=np.float16)
    runs = [runtime.generate(arguments) for _ in range(trials)]
    plans = None
    if compiler == "coreml":
        prefill = next(iter(runtime._prefill_programs.values()))[0]
        decoder = next(iter(runtime._decoder_programs.values()))[0]
        plans = {
            "prefill": prefill.compute_plan_summary(),
            "decode": decoder.compute_plan_summary(),
        }
    return {"runs": runs, "compute_plans": plans}


def run_worker(compiler: str, trials: int) -> Dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            compiler,
            "--trials",
            str(trials),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{compiler} worker failed with {completed.returncode}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    raise RuntimeError(f"{compiler} worker returned no result")


def summarize(runs: List[Dict[str, object]]) -> Dict[str, object]:
    warm = runs[1:] or runs
    return {
        "cold_first_token_seconds": runs[0]["first_token_seconds"],
        "cold_total_seconds": runs[0]["wall_generation_seconds"],
        "warm_first_token_seconds": statistics.median(
            float(run["first_token_seconds"]) for run in warm
        ),
        "warm_decode_tokens_per_second": statistics.median(
            float(run["warm_tokens_per_second"]) for run in warm
        ),
        "warm_total_seconds": statistics.median(
            float(run["wall_generation_seconds"]) for run in warm
        ),
        "peak_device_mib": max(
            (
                float(run["gpu_peak_mib"])
                for run in runs
                if run["gpu_peak_mib"] is not None
            ),
            default=None,
        ),
        "generated_token_ids": runs[-1]["generated_token_ids"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Chomik TinyLlama FP16 on Core ML/ANE and MLX/GPU"
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--worker", choices=("coreml", "mlx"), help=argparse.SUPPRESS
    )
    arguments = parser.parse_args()
    if arguments.trials <= 0:
        parser.error("--trials must be positive")
    if arguments.worker:
        print(
            RESULT_PREFIX
            + json.dumps(worker(arguments.worker, arguments.trials))
        )
        return

    coreml_result = run_worker("coreml", arguments.trials)
    mlx_result = run_worker("mlx", arguments.trials)
    coreml = summarize(coreml_result["runs"])
    mlx = summarize(mlx_result["runs"])
    if coreml["generated_token_ids"] != mlx["generated_token_ids"]:
        raise RuntimeError("Core ML and MLX generated different FP16 token IDs")
    result = {
        "coreml": coreml,
        "mlx": mlx,
        "coreml_compute_plans": coreml_result["compute_plans"],
    }
    if arguments.json:
        print(json.dumps(result, indent=2))
        return
    print("metric,coreml,mlx")
    for key in (
        "cold_first_token_seconds",
        "cold_total_seconds",
        "warm_first_token_seconds",
        "warm_decode_tokens_per_second",
        "warm_total_seconds",
        "peak_device_mib",
    ):
        print(f"{key},{coreml[key]},{mlx[key]}")
    print(json.dumps(result["coreml_compute_plans"], indent=2))


if __name__ == "__main__":
    main()
