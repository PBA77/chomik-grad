from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from chomikgrad import (
    DeviceAdapter,
    Tensor,
    compile_graph,
    get_compiler,
    no_grad,
    verify_greedy_candidates,
)
from chomikgrad.llama import (
    LlamaConfig,
    LlamaDecoderBlock,
    LlamaDecoderStep,
    LlamaForCausalLM,
    _rope_rotations,
    load_safetensors,
    position_selector,
)


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
DRAFT_MODEL_ID = "Felladrin/Llama-68M-Chat-v1"
DRAFT_MODEL_REVISION = "180d584580aa5cf33558d2bce51f1d125e20c7c7"
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
DRAFT_MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json")


def format_chat(prompt: str, system: Optional[str]) -> str:
    parts = []
    if system:
        parts.append(f"<|system|>\n{system}</s>\n")
    parts.append(f"<|user|>\n{prompt}</s>\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def format_draft_chat(prompt: str, system: Optional[str]) -> str:
    system_text = system or "You are a helpful assistant."
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


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


def draft_model_directory(local: Optional[Path]) -> Path:
    if local is not None:
        missing = [
            name for name in DRAFT_MODEL_FILES if not (local / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"missing draft model file: {missing[0]}")
        return local
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "draft model download requires `chomik-grad[llm]`"
        ) from error
    return Path(
        snapshot_download(
            repo_id=DRAFT_MODEL_ID,
            revision=DRAFT_MODEL_REVISION,
            allow_patterns=list(DRAFT_MODEL_FILES),
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


def sample_native_token(
    logits: Any,
    *,
    temperature: float,
    top_k: int,
    rng: np.random.Generator,
    device: DeviceAdapter,
) -> int:
    """Keep greedy selection on device and copy full logits only for sampling."""
    if temperature == 0:
        return device.argmax(logits)
    return sample_token(
        device.to_numpy(logits),
        temperature=temperature,
        top_k=top_k,
        rng=rng,
    )


class TinyLlamaRuntime:
    """Reusable model state for multiple generation requests."""

    def __init__(
        self,
        local: Optional[Path] = None,
        *,
        compiler: str = "mlx",
        dtype: Optional[np.dtype] = None,
    ) -> None:
        try:
            import ml_dtypes
            from tokenizers import Tokenizer
        except ImportError as error:
            raise ImportError(
                "install generation dependencies with "
                "`python -m pip install '.[llm]'` plus the compiler backend"
            ) from error

        selected_compiler = get_compiler(compiler)
        started = time.perf_counter()
        directory = model_directory(local)
        self.download_seconds = time.perf_counter() - started
        self.config = LlamaConfig.from_dict(
            json.loads((directory / "config.json").read_text())
        )
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        loaded_at = time.perf_counter()
        self.device = selected_compiler.device
        self.compiler_name = compiler
        self.dtype = np.dtype(
            dtype
            if dtype is not None
            else getattr(self.device, "inference_dtype", ml_dtypes.bfloat16)
        )
        self.weights = load_safetensors(
            directory / "model.safetensors", self.device, dtype=self.dtype
        )
        self.device.evaluate(
            [
                tensor._node.native_values[self.device.name]
                for tensor in self.weights.values()
            ]
        )
        self.load_seconds = time.perf_counter() - loaded_at
        self._prefill_programs: Dict[Tuple[int, int], Tuple[Any, Any]] = {}
        self._decoder_programs: Dict[int, Tuple[Any, List[Any]]] = {}
        self._draft_prefill_programs: Dict[
            Tuple[int, int, int], Tuple[Any, Any]
        ] = {}
        self._draft_decoder_programs: Dict[
            Tuple[int, int], Tuple[Any, List[Any]]
        ] = {}
        self._verification_programs: Dict[
            Tuple[int, int], Tuple[Any, List[Any]]
        ] = {}
        self._draft_config: Optional[LlamaConfig] = None
        self._draft_weights: Optional[Dict[str, Tensor]] = None
        self._draft_source: Optional[Path] = None
        self.draft_load_seconds = 0.0

    def load_draft(
        self, local: Optional[Path] = None
    ) -> Tuple[LlamaConfig, Dict[str, Tensor]]:
        if self._draft_config is not None and self._draft_weights is not None:
            if local is not None and local != self._draft_source:
                raise ValueError("this runtime already loaded a different draft model")
            return self._draft_config, self._draft_weights
        from tokenizers import Tokenizer

        started = time.perf_counter()
        directory = draft_model_directory(local)
        config = LlamaConfig.from_dict(
            json.loads((directory / "config.json").read_text())
        )
        if config.vocab_size != self.config.vocab_size:
            raise ValueError("draft and target vocabularies have different sizes")
        draft_tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        if draft_tokenizer.get_vocab() != self.tokenizer.get_vocab():
            raise ValueError(
                "draft and target token IDs do not use the same vocabulary"
            )
        weights = load_safetensors(
            directory / "model.safetensors", self.device, dtype=self.dtype
        )
        self.device.evaluate(
            [
                tensor._node.native_values[self.device.name]
                for tensor in weights.values()
            ]
        )
        self._draft_config = config
        self._draft_weights = weights
        self._draft_source = local
        self.draft_load_seconds = time.perf_counter() - started
        return config, weights

    def generate(self, args: argparse.Namespace) -> Dict[str, object]:
        return _generate_loaded(args, self)


def generate(args: argparse.Namespace) -> Dict[str, object]:
    return TinyLlamaRuntime(
        args.model_dir,
        compiler=args.compiler,
        dtype=getattr(args, "dtype", None),
    ).generate(args)


def _run_draft_prefill(
    runtime: TinyLlamaRuntime,
    config: LlamaConfig,
    weights: Mapping[str, Tensor],
    inputs: np.ndarray,
    selector: np.ndarray,
    context: int,
) -> List[Any]:
    key = (config.num_hidden_layers, inputs.shape[1], context)
    cached = runtime._draft_prefill_programs.get(key)
    if cached is None:
        model = LlamaForCausalLM(
            config,
            weights,
            sequence_length=inputs.shape[1],
            dtype=runtime.dtype,
            cache_length=context,
        )
        input_tensor = Tensor(inputs, copy=False)
        outputs = model.prefill(
            input_tensor, Tensor(selector, copy=False)
        )[1:]
        program = compile_graph(
            *outputs,
            compiler=runtime.compiler_name,
            dynamic_inputs=(input_tensor,),
        )
        runtime._draft_prefill_programs[key] = program, input_tensor._node
        bindings = None
    else:
        program, input_node = cached
        bindings = {input_node: runtime.device.array(inputs)}
    return list(program.run(bindings, synchronize=False))


def _run_decoder_step(
    runtime: TinyLlamaRuntime,
    config: LlamaConfig,
    weights: Mapping[str, Tensor],
    cache_values: List[Any],
    token: int,
    position: int,
    context: int,
) -> Tuple[Any, List[Any]]:
    key = (config.num_hidden_layers, context)
    cached = runtime._draft_decoder_programs.get(key)
    inputs = np.asarray([[token]], dtype=np.int32)
    rotation = _rope_rotations(
        1,
        config.head_size,
        config.rope_theta,
        runtime.dtype,
        offset=position,
    )
    attention_mask = np.full(
        (1, 1, 1, context), -1e4, dtype=runtime.dtype
    )
    attention_mask[..., : position + 1] = 0
    write_mask = np.zeros((1, 1, context, 1), dtype=runtime.dtype)
    write_mask[:, :, position, :] = 1
    if cached is None:
        decoder = LlamaDecoderStep(
            config,
            weights,
            cache_length=context,
            dtype=runtime.dtype,
        )
        input_tensor = Tensor(inputs, copy=False)
        rotation_tensor = Tensor(rotation, copy=False)
        attention_tensor = Tensor(attention_mask, copy=False)
        write_tensor = Tensor(write_mask, copy=False)
        position_tensor = Tensor(
            np.asarray(position, dtype=np.int32), copy=False
        )
        cache_tensors = [
            Tensor.from_native(
                value,
                backend=runtime.device.name,
                shape=tuple(value.shape),
                dtype=runtime.dtype,
            )
            for value in cache_values
        ]
        outputs = decoder(
            input_tensor,
            rotation_tensor,
            attention_tensor,
            write_tensor,
            list(zip(cache_tensors[::2], cache_tensors[1::2])),
            position=position_tensor,
        )
        dynamic_tensors = [
            input_tensor,
            rotation_tensor,
            attention_tensor,
            write_tensor,
            position_tensor,
            *cache_tensors,
        ]
        program = compile_graph(
            *outputs,
            compiler=runtime.compiler_name,
            dynamic_inputs=dynamic_tensors,
        )
        dynamic_nodes = [tensor._node for tensor in dynamic_tensors]
        runtime._draft_decoder_programs[key] = program, dynamic_nodes
        bindings = None
    else:
        program, dynamic_nodes = cached
        values = [
            runtime.device.array(inputs),
            runtime.device.array(rotation),
            runtime.device.array(attention_mask),
            runtime.device.array(write_mask),
            runtime.device.array(np.asarray(position, dtype=np.int32)),
            *cache_values,
        ]
        bindings = dict(zip(dynamic_nodes, values))
    native = program.run(bindings, synchronize=False)
    return native[0], list(native[1:])


def _run_verification_block(
    runtime: TinyLlamaRuntime,
    cache_values: List[Any],
    input_tokens: Sequence[int],
    position: int,
    context: int,
) -> Tuple[np.ndarray, List[Any]]:
    block_length = len(input_tokens)
    key = (block_length, context)
    cached = runtime._verification_programs.get(key)
    inputs = np.asarray([input_tokens], dtype=np.int32)
    rotation = _rope_rotations(
        block_length,
        runtime.config.head_size,
        runtime.config.rope_theta,
        runtime.dtype,
        offset=position,
    )
    placement = np.zeros((block_length, context), dtype=runtime.dtype)
    indexes = np.arange(block_length)
    placement[indexes, position + indexes] = 1
    write_mask = placement.sum(axis=0).reshape(1, 1, context, 1)
    attention_mask = np.full(
        (1, 1, block_length, context), -1e4, dtype=runtime.dtype
    )
    for index in range(block_length):
        attention_mask[..., index, : position + index + 1] = 0
    if cached is None:
        decoder = LlamaDecoderBlock(
            runtime.config,
            runtime.weights,
            block_length=block_length,
            cache_length=context,
            dtype=runtime.dtype,
        )
        input_tensor = Tensor(inputs, copy=False)
        rotation_tensor = Tensor(rotation, copy=False)
        attention_tensor = Tensor(attention_mask, copy=False)
        placement_tensor = Tensor(placement, copy=False)
        write_tensor = Tensor(write_mask, copy=False)
        position_tensor = Tensor(
            np.asarray(position, dtype=np.int32), copy=False
        )
        cache_tensors = [
            Tensor.from_native(
                value,
                backend=runtime.device.name,
                shape=tuple(value.shape),
                dtype=runtime.dtype,
            )
            for value in cache_values
        ]
        outputs = decoder(
            input_tensor,
            rotation_tensor,
            attention_tensor,
            placement_tensor,
            write_tensor,
            list(zip(cache_tensors[::2], cache_tensors[1::2])),
            position=position_tensor,
        )
        dynamic_tensors = [
            input_tensor,
            rotation_tensor,
            attention_tensor,
            placement_tensor,
            write_tensor,
            position_tensor,
            *cache_tensors,
        ]
        program = compile_graph(
            *outputs,
            compiler=runtime.compiler_name,
            dynamic_inputs=dynamic_tensors,
        )
        dynamic_nodes = [tensor._node for tensor in dynamic_tensors]
        runtime._verification_programs[key] = program, dynamic_nodes
        bindings = None
    else:
        program, dynamic_nodes = cached
        values = [
            runtime.device.array(inputs),
            runtime.device.array(rotation),
            runtime.device.array(attention_mask),
            runtime.device.array(placement),
            runtime.device.array(write_mask),
            runtime.device.array(np.asarray(position, dtype=np.int32)),
            *cache_values,
        ]
        bindings = dict(zip(dynamic_nodes, values))
    native = program.run(bindings, synchronize=False)
    target_tokens = runtime.device.argmax_last_axis(native[0]).reshape(-1)
    return target_tokens, list(native[1:])


def _generate_speculative_loaded(
    args: argparse.Namespace, runtime: TinyLlamaRuntime
) -> Dict[str, object]:
    config = runtime.config
    if args.temperature != 0:
        raise ValueError("speculative decoding currently requires temperature=0")
    draft_config, draft_weights = runtime.load_draft(args.draft_model_dir)
    rendered_prompt = format_chat(args.prompt, args.system)
    prompt_ids = encode_chat(runtime.tokenizer, rendered_prompt, config.eos_token_id)
    draft_rendered_prompt = format_draft_chat(args.prompt, args.system)
    draft_prompt_ids = runtime.tokenizer.encode(
        draft_rendered_prompt, add_special_tokens=False
    ).ids
    if not prompt_ids:
        raise ValueError("the prompt produced no tokens")
    context = len(prompt_ids) + args.max_new_tokens
    if context > config.max_position_embeddings:
        raise ValueError(
            f"prompt plus output is {context} tokens, model limit is "
            f"{config.max_position_embeddings}"
        )
    draft_context = len(draft_prompt_ids) + args.max_new_tokens
    if draft_context > draft_config.max_position_embeddings:
        raise ValueError(
            f"draft prompt plus output is {draft_context} tokens, model limit is "
            f"{draft_config.max_position_embeddings}"
        )

    model = LlamaForCausalLM(
        config,
        runtime.weights,
        sequence_length=len(prompt_ids),
        dtype=runtime.dtype,
        cache_length=context,
    )
    inputs = np.asarray([prompt_ids], dtype=np.int32)
    selector = position_selector(
        len(prompt_ids) - 1, len(prompt_ids), dtype=runtime.dtype
    )
    runtime.device.reset_peak_memory()
    generation_started = time.perf_counter()
    with no_grad():
        first_started = time.perf_counter()
        prefill_key = (len(prompt_ids), context)
        cached_prefill = runtime._prefill_programs.get(prefill_key)
        if cached_prefill is None:
            input_tensor = Tensor(inputs, copy=False)
            prefill_outputs = model.prefill(
                input_tensor, Tensor(selector, copy=False)
            )
            prefill_program = compile_graph(
                *prefill_outputs,
                compiler=runtime.compiler_name,
                dynamic_inputs=(input_tensor,),
            )
            runtime._prefill_programs[prefill_key] = (
                prefill_program,
                input_tensor._node,
            )
            prefill_bindings = None
        else:
            prefill_program, input_node = cached_prefill
            prefill_bindings = {input_node: runtime.device.array(inputs)}
        target_native = prefill_program.run(
            prefill_bindings, synchronize=False
        )
        token = runtime.device.argmax(target_native[0])
        first_token_seconds = time.perf_counter() - first_started
        target_caches = list(target_native[1:])
        generated = [token]
        tokens = list(prompt_ids) + [token]

        draft_inputs = np.asarray([draft_prompt_ids], dtype=np.int32)
        draft_selector = position_selector(
            len(draft_prompt_ids) - 1,
            len(draft_prompt_ids),
            dtype=runtime.dtype,
        )
        draft_caches = _run_draft_prefill(
            runtime,
            draft_config,
            draft_weights,
            draft_inputs,
            draft_selector,
            draft_context,
        )
        decode_started = time.perf_counter()
        accepted = 0
        proposed = 0
        verification_calls = 0
        draft_calls = 0
        while len(generated) < args.max_new_tokens and token != config.eos_token_id:
            block_limit = min(
                args.speculative_tokens,
                args.max_new_tokens - len(generated),
            )
            draft_tokens = []
            draft_token = token
            draft_position = len(draft_prompt_ids) + len(generated) - 1
            for _ in range(block_limit):
                draft_logits, draft_caches = _run_decoder_step(
                    runtime,
                    draft_config,
                    draft_weights,
                    draft_caches,
                    draft_token,
                    draft_position,
                    draft_context,
                )
                draft_token = runtime.device.argmax(draft_logits)
                draft_tokens.append(draft_token)
                draft_calls += 1
                draft_position += 1
                if draft_token == config.eos_token_id:
                    break

            target_inputs = [token, *draft_tokens[:-1]]
            target_tokens, target_caches = _run_verification_block(
                runtime,
                target_caches,
                target_inputs,
                len(tokens) - 1,
                context,
            )
            decision = verify_greedy_candidates(draft_tokens, target_tokens)
            emitted = list(decision.emitted_tokens)
            tokens.extend(emitted)
            generated.extend(emitted)
            token = emitted[-1]
            accepted += decision.accepted_draft_tokens
            proposed += len(draft_tokens)
            verification_calls += 1

    wall_generation_seconds = time.perf_counter() - generation_started
    decode_seconds = time.perf_counter() - decode_started
    response = runtime.tokenizer.decode(generated, skip_special_tokens=True)
    peak_memory = runtime.device.peak_memory_bytes()
    decode_tokens = max(0, len(generated) - 1)
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "parameters": model.parameter_count,
        "dtype": runtime.dtype.name,
        "compiler": runtime.compiler_name,
        "prompt": args.prompt,
        "rendered_prompt": rendered_prompt,
        "draft_rendered_prompt": draft_rendered_prompt,
        "prompt_tokens": len(prompt_ids),
        "draft_prompt_tokens": len(draft_prompt_ids),
        "generated_tokens": len(generated),
        "generated_token_ids": generated,
        "response": response,
        "download_seconds": runtime.download_seconds,
        "load_seconds": runtime.load_seconds,
        "draft_load_seconds": runtime.draft_load_seconds,
        "first_token_seconds": first_token_seconds,
        "warm_token_median_seconds": (
            decode_seconds / decode_tokens if decode_tokens else None
        ),
        "warm_tokens_per_second": (
            decode_tokens / decode_seconds if decode_tokens else None
        ),
        "total_generation_seconds": wall_generation_seconds,
        "wall_generation_seconds": wall_generation_seconds,
        "gpu_peak_mib": (
            peak_memory / (1024 * 1024) if peak_memory is not None else None
        ),
        "fixed_context_tokens": context,
        "kv_cache": True,
        "speculative_tokens": args.speculative_tokens,
        "draft_model": (
            str(runtime._draft_source)
            if runtime._draft_source is not None
            else DRAFT_MODEL_ID
        ),
        "draft_revision": DRAFT_MODEL_REVISION,
        "draft_parameters": sum(
            int(np.prod(weight.shape)) for weight in draft_weights.values()
        ),
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": proposed,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "target_verification_calls": verification_calls,
        "draft_decode_calls": draft_calls,
    }


def _generate_loaded(
    args: argparse.Namespace, runtime: TinyLlamaRuntime
) -> Dict[str, object]:
    if getattr(args, "speculative_tokens", 0):
        return _generate_speculative_loaded(args, runtime)
    config = runtime.config
    tokenizer = runtime.tokenizer
    weights = runtime.weights
    dtype = runtime.dtype
    device = runtime.device
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
    rng = np.random.default_rng(args.seed)
    tokens: List[int] = list(prompt_ids)
    generated: List[int] = []
    token_seconds: List[float] = []

    device.reset_peak_memory()
    generation_started = time.perf_counter()
    with no_grad():
        inputs = np.asarray(tokens, dtype=np.int32).reshape(1, -1)
        selector = position_selector(
            len(prompt_ids) - 1,
            len(prompt_ids),
            dtype=dtype,
        )
        step_started = time.perf_counter()
        prefill_key = (len(prompt_ids), context)
        cached_prefill = runtime._prefill_programs.get(prefill_key)
        if cached_prefill is None:
            input_tensor = Tensor(inputs, copy=False)
            prefill_outputs = model.prefill(
                input_tensor, Tensor(selector, copy=False)
            )
            prefill_program = compile_graph(
                *prefill_outputs,
                compiler=runtime.compiler_name,
                dynamic_inputs=(input_tensor,),
            )
            runtime._prefill_programs[prefill_key] = (
                prefill_program,
                input_tensor._node,
            )
            prefill_bindings = None
        else:
            prefill_program, input_node = cached_prefill
            prefill_bindings = {input_node: device.array(inputs)}
        native_outputs = prefill_program.run(
            prefill_bindings,
            synchronize=False,
        )
        token = sample_native_token(
            native_outputs[0],
            temperature=args.temperature,
            top_k=args.top_k,
            rng=rng,
            device=device,
        )
        token_seconds.append(time.perf_counter() - step_started)
        tokens.append(token)
        generated.append(token)
        cache_values = list(native_outputs[1:])
        cached_decoder = runtime._decoder_programs.get(context)
        if cached_decoder is None:
            decoder_program = None
            dynamic_nodes = []
        else:
            decoder_program, dynamic_nodes = cached_decoder

        for _ in range(1, args.max_new_tokens):
            if token == config.eos_token_id:
                break
            position = len(tokens) - 1
            inputs = np.asarray([[token]], dtype=np.int32)
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
            if decoder_program is None:
                input_tensor = Tensor(inputs, copy=False)
                rotation_tensor = Tensor(rotation, copy=False)
                attention_tensor = Tensor(attention_mask, copy=False)
                write_tensor = Tensor(write_mask, copy=False)
                position_tensor = Tensor(
                    np.asarray(position, dtype=np.int32), copy=False
                )
                cache_tensors = [
                    Tensor.from_native(
                        value,
                        backend=device.name,
                        shape=tuple(value.shape),
                        dtype=dtype,
                    )
                    for value in cache_values
                ]
                caches = list(zip(cache_tensors[::2], cache_tensors[1::2]))
                outputs = decoder(
                    input_tensor,
                    rotation_tensor,
                    attention_tensor,
                    write_tensor,
                    caches,
                    position=position_tensor,
                )
                dynamic_tensors = [
                    input_tensor,
                    rotation_tensor,
                    attention_tensor,
                    write_tensor,
                    position_tensor,
                    *cache_tensors,
                ]
                decoder_program = compile_graph(
                    *outputs,
                    compiler=runtime.compiler_name,
                    dynamic_inputs=dynamic_tensors,
                )
                dynamic_nodes = [tensor._node for tensor in dynamic_tensors]
                runtime._decoder_programs[context] = (
                    decoder_program,
                    dynamic_nodes,
                )
                bindings = None
            else:
                values = [
                    device.array(inputs),
                    device.array(rotation),
                    device.array(attention_mask),
                    device.array(write_mask),
                    device.array(np.asarray(position, dtype=np.int32)),
                    *cache_values,
                ]
                bindings = dict(zip(dynamic_nodes, values))
            native_outputs = decoder_program.run(
                bindings,
                synchronize=False,
            )
            token = sample_native_token(
                native_outputs[0],
                temperature=args.temperature,
                top_k=args.top_k,
                rng=rng,
                device=device,
            )
            token_seconds.append(time.perf_counter() - step_started)
            tokens.append(token)
            generated.append(token)
            cache_values = list(native_outputs[1:])

    response = tokenizer.decode(generated, skip_special_tokens=True)
    wall_generation_seconds = time.perf_counter() - generation_started
    warm = token_seconds[1:]
    peak_memory = device.peak_memory_bytes()
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "parameters": model.parameter_count,
        "dtype": runtime.dtype.name,
        "compiler": runtime.compiler_name,
        "prompt": args.prompt,
        "rendered_prompt": rendered_prompt,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated),
        "generated_token_ids": generated,
        "response": response,
        "download_seconds": runtime.download_seconds,
        "load_seconds": runtime.load_seconds,
        "first_token_seconds": token_seconds[0],
        "warm_token_median_seconds": statistics.median(warm) if warm else None,
        "warm_tokens_per_second": 1 / statistics.median(warm) if warm else None,
        "total_generation_seconds": sum(token_seconds),
        "wall_generation_seconds": wall_generation_seconds,
        "gpu_peak_mib": (
            peak_memory / (1024 * 1024) if peak_memory is not None else None
        ),
        "fixed_context_tokens": context,
        "kv_cache": True,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with real TinyLlama weights on Chomik"
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
    parser.add_argument(
        "--speculative-tokens",
        type=int,
        default=0,
        help=(
            "experimental draft tokens per greedy target block; batched BF16 "
            "can differ numerically; 0 disables it"
        ),
    )
    parser.add_argument(
        "--draft-model-dir",
        type=Path,
        help=f"local {DRAFT_MODEL_ID} snapshot; defaults to a pinned download",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--compiler", default="mlx")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        help="defaults to BF16 on MLX and FP16 on Core ML",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if args.top_k < 0:
        parser.error("--top-k cannot be negative")
    if args.speculative_tokens < 0:
        parser.error("--speculative-tokens cannot be negative")
    if args.speculative_tokens and args.temperature != 0:
        parser.error("speculative decoding requires --temperature 0")
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
    if result["gpu_peak_mib"] is not None:
        print(f"GPU peak: {result['gpu_peak_mib']:.1f} MiB")
    if result.get("speculative_tokens"):
        print(
            "Draft acceptance: "
            f"{100 * result['acceptance_rate']:.1f}% "
            f"({result['accepted_draft_tokens']}/"
            f"{result['proposed_draft_tokens']})"
        )


if __name__ == "__main__":
    main()
