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
RESULT_PREFIX = "CHOMIK_SPECULATIVE_RESULT="
CASES = (
    "What is the capital of France? Answer in one short sentence.",
    "Write a Python function that returns the factorial of an integer.",
    "List five primary colors, one per line.",
    "Write a detailed story about a robot exploring an ancient city.",
    "Explain lazy execution in one short sentence.",
    "What is 17 times 23? Give only the result.",
    "Translate to Polish: The weather is beautiful today.",
    "Write a four-line poem about the Moon.",
    "Name the three largest planets in the Solar System.",
    "Count from one to ten using words separated by commas.",
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def worker(speculative_tokens: int, trials: int) -> List[Dict[str, object]]:
    from examples.generate_tinyllama import TinyLlamaRuntime

    runtime = TinyLlamaRuntime()
    results = []
    for prompt in CASES:
        arguments = SimpleNamespace(
            prompt=prompt,
            system=None,
            max_new_tokens=32,
            temperature=0.0,
            top_k=50,
            seed=0,
            model_dir=None,
            json=True,
            speculative_tokens=speculative_tokens,
            draft_model_dir=None,
        )
        runtime.generate(arguments)
        runs = []
        for _ in range(trials):
            started = time.perf_counter()
            output = runtime.generate(arguments)
            runs.append(time.perf_counter() - started)
        results.append(
            {
                "prompt": prompt,
                "seconds": statistics.median(runs),
                "generated_tokens": output["generated_tokens"],
                "generated_token_ids": output["generated_token_ids"],
                "acceptance_rate": output.get("acceptance_rate"),
            }
        )
    return results


def run_worker(mode: str, speculative_tokens: int, trials: int):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            mode,
            "--speculative-tokens",
            str(speculative_tokens),
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
            f"{mode} worker failed:\n{completed.stdout}\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    raise RuntimeError(f"{mode} worker returned no result")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure exact-token coverage and speed of speculative MLX decode"
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--speculative-tokens", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--worker", choices=("baseline", "speculative"))
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("trials must be positive")
    if args.speculative_tokens <= 0:
        parser.error("speculative tokens must be positive")
    if args.worker:
        count = args.speculative_tokens if args.worker == "speculative" else 0
        print(RESULT_PREFIX + json.dumps(worker(count, args.trials)))
        return

    baseline = run_worker("baseline", args.speculative_tokens, args.trials)
    speculative = run_worker("speculative", args.speculative_tokens, args.trials)
    cases = []
    for base, spec in zip(baseline, speculative):
        same = base["generated_token_ids"] == spec["generated_token_ids"]
        cases.append(
            {
                "prompt": base["prompt"],
                "baseline_seconds": base["seconds"],
                "speculative_seconds": spec["seconds"],
                "speedup": base["seconds"] / spec["seconds"],
                "same_tokens": same,
                "acceptance_rate": spec["acceptance_rate"],
            }
        )
    result = {
        "speculative_tokens": args.speculative_tokens,
        "matching_cases": sum(case["same_tokens"] for case in cases),
        "faster_cases": sum(case["speedup"] > 1 for case in cases),
        "cases": cases,
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print("case,same_tokens,acceptance,speedup")
    for index, case in enumerate(cases, 1):
        print(
            f"{index},{case['same_tokens']},{case['acceptance_rate']:.3f},"
            f"{case['speedup']:.3f}"
        )
    print(
        f"matching={result['matching_cases']}/{len(cases)},"
        f"faster={result['faster_cases']}/{len(cases)}"
    )


if __name__ == "__main__":
    main()
