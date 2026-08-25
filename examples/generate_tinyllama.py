from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from chomikgrad import Tensor, compile_graph, no_grad
from chomikgrad.llama import (
    LlamaConfig,
    LlamaDecoderStep,
    LlamaForCausalLM,
    _rope_rotations,
    load_mlx_safetensors,
    position_selector,
    token_one_hot,
)


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def format_chat(prompt: str, system: Optional[str]) -> str:
    parts = []
    if system:
        parts.append(f"<|system|>\n{system}</s>\n")
    parts.append(f"<|user|>\n{prompt}</s>\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def encode_chat(
    tokenizer: Any, rendered_prompt: str, eos_token_id: int
) -> List[int]:
    encoded = tokenizer.encode(rendered_prompt, add_special_tokens=False).ids
    dummy_prefix_id = tokenizer.token_to_id("▁")
    # tokenizer.json applies SentencePiece's dummy prefix after a recognized
    # special token. The official non-legacy LlamaTokenizer removes it.
    return [
        token
        for index, token in enumerate(encoded)
        if not (
            token == dummy_prefix_id
            and index > 0
            and encoded[index - 1] == eos_token_id
        )
    ]


def model_directory(local: Optional[Path]) -> Path:
    if local is not None:
        missing = [name for name in MODEL_FILES if not (local / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing model file: {missing[0]}")
        return local
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "TinyLlama download requires `chomik-grad[llm]`"
        ) from error
    return Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(MODEL_FILES),
        )
    )


def sample_token(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int,
    rng: np.random.Generator,
) -> int:
    values = np.asarray(logits[0], dtype=np.float64)
    if temperature == 0:
        return int(values.argmax())
    values /= temperature
    if top_k > 0 and top_k < values.size:
        indexes = np.argpartition(values, -top_k)[-top_k:]
        filtered = np.full_like(values, -np.inf)
        filtered[indexes] = values[indexes]
        values = filtered
    values -= np.max(values)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    return int(rng.choice(values.size, p=probabilities))


def generate(args: argparse.Namespace) -> Dict[str, object]:
    try:
        import ml_dtypes
        import mlx.core as mx
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ImportError(
            "install generation dependencies with `python -m pip install '.[llm,mlx]'`"
        ) from error

    started = time.perf_counter()
    directory = model_directory(args.model_dir)
    download_seconds = time.perf_counter() - started
    config = LlamaConfig.from_dict(
        json.loads((directory / "config.json").read_text())
    )
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    rendered_prompt = format_chat(args.prompt, args.system)
    prompt_ids = encode_chat(tokenizer, rendered_prompt, config.eos_token_id)
    if not prompt_ids:
        raise ValueError("the prompt produced no tokens")
    context = len(prompt_ids) + args.max_new_tokens
    if context > config.max_position_embeddings:
        raise ValueError(
            f"prompt plus output is {context} tokens, model limit is "
            f"{config.max_position_embeddings}"
        )

    loaded_at = time.perf_counter()
    weights = load_mlx_safetensors(directory / "model.safetensors")
    dtype = np.dtype(ml_dtypes.bfloat16)
    model = LlamaForCausalLM(
        config,
        weights,
        sequence_length=len(prompt_ids),
        dtype=dtype,
        cache_length=context,
    )
    decoder = LlamaDecoderStep(
        config,
        weights,
        cache_length=context,
        dtype=dtype,
    )
    load_seconds = time.perf_counter() - loaded_at
    rng = np.random.default_rng(args.seed)
    tokens: List[int] = list(prompt_ids)
    generated: List[int] = []
    token_seconds: List[float] = []

    mx.reset_peak_memory()
    with no_grad():
        inputs = token_one_hot(
            tokens,
            len(prompt_ids),
            config.vocab_size,
            fill_token_id=config.eos_token_id,
            dtype=dtype,
        )
        selector = position_selector(
            len(prompt_ids) - 1,
            len(prompt_ids),
            dtype=dtype,
        )
        step_started = time.perf_counter()
        prefill_outputs = model.prefill(
            Tensor(inputs, copy=False), Tensor(selector, copy=False)
        )
        prefill_program = compile_graph(*prefill_outputs, compiler="mlx")
        native_outputs = prefill_program.run_native()  # type: ignore[attr-defined]
        logits = np.array(native_outputs[0].astype(mx.float32))
        token = sample_token(
            logits,
            temperature=args.temperature,
            top_k=args.top_k,
            rng=rng,
        )
        token_seconds.append(time.perf_counter() - step_started)
        tokens.append(token)
        generated.append(token)
        cache_tensors = [
            Tensor.from_native(
                value,
                backend="mlx",
                shape=tuple(value.shape),
                dtype=dtype,
            )
            for value in native_outputs[1:]
        ]
        caches = list(zip(cache_tensors[::2], cache_tensors[1::2]))

        for _ in range(1, args.max_new_tokens):
            if token == config.eos_token_id:
                break
            position = len(tokens) - 1
            inputs = token_one_hot(
                [token],
                1,
                config.vocab_size,
                fill_token_id=config.eos_token_id,
                dtype=dtype,
            )
            rotation = _rope_rotations(
                1,
                config.head_size,
                config.rope_theta,
                dtype,
                offset=position,
            )
            attention_mask = np.full(
                (1, 1, 1, context), -1e4, dtype=dtype
            )
            attention_mask[..., : position + 1] = 0
            write_mask = np.zeros((1, 1, context, 1), dtype=dtype)
            write_mask[:, :, position, :] = 1
            step_started = time.perf_counter()
            outputs = decoder(
                Tensor(inputs, copy=False),
                Tensor(rotation, copy=False),
                Tensor(attention_mask, copy=False),
                Tensor(write_mask, copy=False),
                caches,
            )
            program = compile_graph(*outputs, compiler="mlx")
            native_outputs = program.run_native()  # type: ignore[attr-defined]
            logits = np.array(native_outputs[0].astype(mx.float32))
            token = sample_token(
                logits,
                temperature=args.temperature,
                top_k=args.top_k,
                rng=rng,
            )
            token_seconds.append(time.perf_counter() - step_started)
            tokens.append(token)
            generated.append(token)
            cache_tensors = [
                Tensor.from_native(
                    value,
                    backend="mlx",
                    shape=tuple(value.shape),
                    dtype=dtype,
                )
                for value in native_outputs[1:]
            ]
            caches = list(zip(cache_tensors[::2], cache_tensors[1::2]))

    response = tokenizer.decode(generated, skip_special_tokens=True)
    warm = token_seconds[1:]
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "parameters": model.parameter_count,
        "dtype": "bfloat16",
        "prompt": args.prompt,
        "rendered_prompt": rendered_prompt,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated),
        "generated_token_ids": generated,
        "response": response,
        "download_seconds": download_seconds,
        "load_seconds": load_seconds,
        "first_token_seconds": token_seconds[0],
        "warm_token_median_seconds": statistics.median(warm) if warm else None,
        "warm_tokens_per_second": 1 / statistics.median(warm) if warm else None,
        "total_generation_seconds": sum(token_seconds),
        "gpu_peak_mib": mx.get_peak_memory() / (1024 * 1024),
        "fixed_context_tokens": context,
        "kv_cache": True,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with real TinyLlama weights on Chomik + Metal"
    )
    parser.add_argument(
        "--prompt",
        default="What is the capital of France? Answer in one short sentence.",
    )
    parser.add_argument("--system")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 selects deterministic greedy decoding",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if args.top_k < 0:
        parser.error("--top-k cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    result = generate(args)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print(f"Model: {result['model']} @ {result['revision'][:12]}")
    print(f"Parameters: {result['parameters']:,} ({result['dtype']})")
    print(f"Prompt tokens: {result['prompt_tokens']}")
    print(f"Response: {result['response']}")
    print(f"First token: {result['first_token_seconds']:.3f} s")
    if result["warm_tokens_per_second"] is not None:
        print(f"Warm decode: {result['warm_tokens_per_second']:.2f} token/s")
    print(f"GPU peak: {result['gpu_peak_mib']:.1f} MiB")


if __name__ == "__main__":
    main()
