"""Transformers-backed model adapters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from ..config import ModelConfig
from ..prompting import content_with_images, merge_system_and_question
from .base import BaseTransformersAdapter

try:
    from peft import PeftConfig, PeftModel
except Exception:  # noqa: BLE001
    PeftConfig = None
    PeftModel = None

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # noqa: BLE001
    process_vision_info = None


def _import_transformers() -> Any:
    import transformers

    return transformers


def _detect_device(config: ModelConfig) -> str:
    requested = config.runtime.device
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _resolve_device_map(config: ModelConfig) -> str | None:
    value = config.runtime.device_map.strip()
    if value.lower() in {"", "none"}:
        return None
    return value


def _pick_torch_dtype(config: ModelConfig) -> torch.dtype:
    dtype = (config.runtime.dtype or "auto").lower()
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    if torch.cuda.is_available() and config.runtime.prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _detect_model_device(model: Any) -> torch.device:
    if hasattr(model, "device") and isinstance(model.device, torch.device):
        return model.device
    try:
        return next(model.parameters()).device
    except Exception:  # noqa: BLE001
        return torch.device("cpu")


def _move_value_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
    return value


def _move_inputs_to_device(inputs: Any, device: torch.device) -> dict[str, Any]:
    if hasattr(inputs, "to"):
        try:
            moved = inputs.to(device)
            if isinstance(moved, dict):
                return dict(moved)
        except Exception:  # noqa: BLE001
            pass
    if isinstance(inputs, dict):
        return _move_value_to_device(inputs, device)
    if hasattr(inputs, "items"):
        return _move_value_to_device(dict(inputs.items()), device)
    raise RuntimeError(f"Unsupported inputs type: {type(inputs)!r}")


def _trim_generated_ids(output_ids: torch.Tensor, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
    if input_ids is None or output_ids.ndim != 2 or input_ids.ndim != 2:
        return output_ids
    prefix_len = input_ids.shape[1]
    if output_ids.shape[1] <= prefix_len:
        return output_ids
    return output_ids[:, prefix_len:]


def _decode_generated(decoder: Any, output_ids: torch.Tensor, input_ids: Optional[torch.Tensor]) -> str:
    generated_ids = _trim_generated_ids(output_ids, input_ids)
    if torch.is_tensor(generated_ids):
        generated_ids = generated_ids.detach().cpu()
    if hasattr(decoder, "batch_decode"):
        text = decoder.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()
    if hasattr(decoder, "decode"):
        return decoder.decode(generated_ids[0], skip_special_tokens=True).strip()
    return str(generated_ids.tolist()).strip()


def _build_generation_kwargs(config: ModelConfig) -> dict[str, Any]:
    generation = config.generation
    runtime = config.runtime
    kwargs: dict[str, Any] = {
        "max_new_tokens": generation.max_new_tokens,
    }
    do_sample = runtime.do_sample
    if do_sample is None:
        do_sample = generation.temperature > 0
    kwargs["do_sample"] = bool(do_sample)
    if kwargs["do_sample"]:
        kwargs["temperature"] = generation.temperature if generation.temperature > 0 else 1.0
        kwargs["top_p"] = generation.top_p
        if generation.top_k > 0:
            kwargs["top_k"] = generation.top_k
    if generation.repetition_penalty and generation.repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = generation.repetition_penalty
    if runtime.num_beams and runtime.num_beams > 1:
        kwargs["num_beams"] = runtime.num_beams
        kwargs["do_sample"] = False
    return kwargs


def _apply_chat_template(processor: Any, messages: list[dict[str, Any]]) -> str:
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raise RuntimeError("Processor/tokenizer does not support apply_chat_template")


def _load_causal_lm(config: ModelConfig, model_name: str, logger: logging.Logger) -> Any:
    transformers = _import_transformers()
    device = _detect_device(config)
    device_map = _resolve_device_map(config)
    torch_dtype = _pick_torch_dtype(config)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.runtime.trust_remote_code,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if config.runtime.revision:
        model_kwargs["revision"] = config.runtime.revision
    if config.runtime.attn_implementation:
        model_kwargs["attn_implementation"] = config.runtime.attn_implementation
    if device_map:
        model_kwargs["device_map"] = device_map

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError as exc:
        retry_kwargs = dict(model_kwargs)
        for key in ("attn_implementation", "low_cpu_mem_usage"):
            retry_kwargs.pop(key, None)
        logger.warning("Retrying model load without unsupported kwargs | model=%s | error=%s", model_name, exc)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **retry_kwargs)
    except Exception as exc:  # noqa: BLE001
        if device_map and "accelerate" in str(exc).lower():
            retry_kwargs = dict(model_kwargs)
            retry_kwargs.pop("device_map", None)
            logger.warning("Retrying model load without device_map | model=%s", model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **retry_kwargs)
        else:
            raise

    if not device_map and hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _load_processor_and_tokenizer(
    config: ModelConfig,
    sources: Sequence[str],
    logger: logging.Logger,
) -> tuple[Any, Any]:
    transformers = _import_transformers()
    processor = None
    tokenizer = None
    last_error: Exception | None = None

    for source in sources:
        if not source:
            continue
        processor_kwargs: dict[str, Any] = {"trust_remote_code": config.runtime.trust_remote_code}
        if config.runtime.use_fast_processor is not None:
            processor_kwargs["use_fast"] = config.runtime.use_fast_processor
        try:
            processor = transformers.AutoProcessor.from_pretrained(source, **processor_kwargs)
            tokenizer = getattr(processor, "tokenizer", None)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Processor load failed | source=%s | error=%s", source, exc)

    for source in sources:
        if not source or tokenizer is not None:
            continue
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                source,
                trust_remote_code=config.runtime.trust_remote_code,
                use_fast=config.runtime.use_fast_tokenizer,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Tokenizer load failed | source=%s | error=%s", source, exc)

    if processor is None or tokenizer is None:
        raise RuntimeError(f"Failed to load processor/tokenizer from {list(sources)}: {last_error}")
    return processor, tokenizer


class QwenTransformersAdapter(BaseTransformersAdapter):
    def __init__(self, config: ModelConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self.model_device = torch.device("cpu")
        self.generation_kwargs = _build_generation_kwargs(config)

    def _resolve_base_model(self) -> str:
        if self.config.base_model:
            return self.config.base_model
        if self.config.adapter_model and PeftConfig is not None:
            try:
                peft_config = PeftConfig.from_pretrained(self.config.adapter_model)
                candidate = getattr(peft_config, "base_model_name_or_path", None)
                if candidate:
                    return str(candidate)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to infer base model from adapter config | error=%s", exc)
        return self.config.model

    def load(self) -> None:
        base_model = self._resolve_base_model()
        model = _load_causal_lm(self.config, base_model, self.logger)

        if self.config.adapter_model:
            if PeftModel is None:
                raise RuntimeError("peft is required when adapter_model is set")
            model = PeftModel.from_pretrained(model, self.config.adapter_model, is_trainable=False)
            if self.config.merge_adapter:
                model = model.merge_and_unload(progressbar=False)

        self.model = model
        self.model_device = _detect_model_device(model)

        sources = []
        if self.config.adapter_model:
            sources.append(self.config.adapter_model)
        sources.append(base_model)
        self.processor, self.tokenizer = _load_processor_and_tokenizer(self.config, sources, self.logger)

    def _build_inputs(
        self,
        system_prompt: str,
        question: str,
        images: Sequence[Any],
        labels: Sequence[str],
        labeled_images: bool,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_with_images(question, images, labels, labeled_images)},
        ]
        text = _apply_chat_template(self.processor, messages)

        if process_vision_info is not None:
            try:
                result = process_vision_info(messages)
                if isinstance(result, tuple) and len(result) >= 2:
                    image_inputs, video_inputs = result[0], result[1]
                    return dict(
                        self.processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                    )
            except Exception:  # noqa: BLE001
                pass

        candidates: list[Any] = [list(images)]
        if len(images) > 1:
            candidates.append([list(images)])
        for payload in candidates:
            try:
                return dict(
                    self.processor(
                        text=[text],
                        images=payload,
                        padding=True,
                        return_tensors="pt",
                    )
                )
            except Exception:  # noqa: BLE001
                continue
        raise RuntimeError("Failed to construct Qwen multimodal inputs")

    def predict(
        self,
        system_prompt: str,
        question: str,
        images: Sequence[Any],
        labels: Sequence[str],
        labeled_images: bool,
    ) -> str:
        model_inputs = self._build_inputs(system_prompt, question, images, labels, labeled_images)
        batch = _move_inputs_to_device(model_inputs, self.model_device)
        with torch.inference_mode():
            output_ids = self.model.generate(**batch, **self.generation_kwargs)
        input_ids = batch.get("input_ids")
        decoder = self.tokenizer or self.processor
        return _decode_generated(decoder, output_ids, input_ids)


class GLMTransformersAdapter(BaseTransformersAdapter):
    def __init__(self, config: ModelConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self.model_device = torch.device("cpu")
        self.generation_kwargs = _build_generation_kwargs(config)
        self.selected_builder: str | None = None

    def load(self) -> None:
        transformers = _import_transformers()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model,
            trust_remote_code=self.config.runtime.trust_remote_code,
            use_fast=self.config.runtime.use_fast_tokenizer,
        )
        try:
            processor_kwargs: dict[str, Any] = {"trust_remote_code": self.config.runtime.trust_remote_code}
            if self.config.runtime.use_fast_processor is not None:
                processor_kwargs["use_fast"] = self.config.runtime.use_fast_processor
            self.processor = transformers.AutoProcessor.from_pretrained(self.config.model, **processor_kwargs)
        except Exception:  # noqa: BLE001
            self.processor = None
        self.model = _load_causal_lm(self.config, self.config.model, self.logger)
        self.model_device = _detect_model_device(self.model)

    def _tokenizer_chat_encode(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs = {"add_generation_prompt": True, "tokenize": True, "return_tensors": "pt"}
        try:
            encoded = self.tokenizer.apply_chat_template(messages, return_dict=True, **kwargs)
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(encoded, dict):
            return dict(encoded)
        if torch.is_tensor(encoded):
            return {"input_ids": encoded}
        if hasattr(encoded, "items"):
            return dict(encoded.items())
        raise RuntimeError(f"Unsupported tokenizer.apply_chat_template output: {type(encoded)!r}")

    def _build_with_tokenizer_image(self, prompt: str, images: Sequence[Any], labels: Sequence[str], labeled: bool) -> dict[str, Any]:
        payload: Any = list(images) if len(images) > 1 else images[0]
        user_content = prompt if not (labeled and labels) else f"{prompt}\nImage order: {', '.join(labels)}."
        messages = [{"role": "user", "content": user_content, "image": payload}]
        return self._tokenizer_chat_encode(messages)

    def _build_with_tokenizer_images(self, prompt: str, images: Sequence[Any], labels: Sequence[str], labeled: bool) -> dict[str, Any]:
        user_content = prompt if not (labeled and labels) else f"{prompt}\nImage order: {', '.join(labels)}."
        messages = [{"role": "user", "content": user_content, "images": list(images)}]
        return self._tokenizer_chat_encode(messages)

    def _build_with_tokenizer_content(self, prompt: str, images: Sequence[Any], labels: Sequence[str], labeled: bool) -> dict[str, Any]:
        messages = [{"role": "user", "content": content_with_images(prompt, images, labels, labeled)}]
        return self._tokenizer_chat_encode(messages)

    def _build_with_processor(self, prompt: str, images: Sequence[Any], labels: Sequence[str], labeled: bool) -> dict[str, Any]:
        if self.processor is None or not hasattr(self.processor, "apply_chat_template"):
            raise RuntimeError("Processor unavailable")
        messages = [{"role": "user", "content": content_with_images(prompt, images, labels, labeled)}]
        templated = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for payload in (list(images), [list(images)]):
            try:
                encoded = self.processor(
                    text=[templated],
                    images=payload,
                    padding=True,
                    return_tensors="pt",
                )
                out = dict(encoded)
                out.pop("token_type_ids", None)
                return out
            except Exception:  # noqa: BLE001
                continue
        raise RuntimeError("Processor input path failed")

    def _build_with_manual_prompt(self, prompt: str, images: Sequence[Any]) -> dict[str, Any]:
        if self.processor is None:
            raise RuntimeError("Processor unavailable")
        image_tokens = "".join("<|begin_of_image|><|image|><|end_of_image|>\n" for _ in images)
        manual_prompt = (
            "[gMASK]<sop><|system|>\nYou are a helpful assistant."
            "<|user|>\n"
            f"{image_tokens}{prompt}"
            "<|assistant|>assistant\n"
        )
        encoded = self.processor(
            text=[manual_prompt],
            images=list(images),
            padding=True,
            return_tensors="pt",
        )
        out = dict(encoded)
        out.pop("token_type_ids", None)
        return out

    def _build_model_inputs(
        self,
        prompt: str,
        images: Sequence[Any],
        labels: Sequence[str],
        labeled_images: bool,
    ) -> dict[str, Any]:
        builders: list[tuple[str, Any]] = []
        if self.processor is not None:
            builders.extend(
                [
                    ("processor", self._build_with_processor),
                    ("manual", lambda p, i, _l, _b: self._build_with_manual_prompt(p, i)),
                ]
            )
        if hasattr(self.tokenizer, "apply_chat_template"):
            builders.extend(
                [
                    ("tokenizer_image", self._build_with_tokenizer_image),
                    ("tokenizer_images", self._build_with_tokenizer_images),
                    ("tokenizer_content", self._build_with_tokenizer_content),
                ]
            )

        if self.selected_builder:
            for name, fn in builders:
                if name != self.selected_builder:
                    continue
                try:
                    return fn(prompt, images, labels, labeled_images)
                except Exception:  # noqa: BLE001
                    self.selected_builder = None
                    break

        errors: list[str] = []
        for name, fn in builders:
            try:
                encoded = fn(prompt, images, labels, labeled_images)
                self.selected_builder = name
                self.logger.info("GLM input builder selected: %s", name)
                return encoded
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
        raise RuntimeError("All GLM input builders failed: " + " | ".join(errors))

    def _chat_fallback(self, prompt: str, images: Sequence[Any]) -> str:
        if not hasattr(self.model, "chat"):
            raise RuntimeError("Model has no chat() fallback")
        payload: Any = list(images) if len(images) > 1 else images[0]
        messages = [{"role": "user", "content": prompt}]
        attempts = [
            lambda: self.model.chat(image=payload, msgs=messages, tokenizer=self.tokenizer),
            lambda: self.model.chat(payload, messages, self.tokenizer),
            lambda: self.model.chat(images=list(images), msgs=messages, tokenizer=self.tokenizer),
            lambda: self.model.chat(msgs=messages, tokenizer=self.tokenizer),
        ]
        last_error: Exception | None = None
        for attempt in attempts:
            try:
                output = attempt()
                if isinstance(output, tuple):
                    output = output[0]
                return str(output).strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"GLM chat fallback failed: {last_error}")

    def predict(
        self,
        system_prompt: str,
        question: str,
        images: Sequence[Any],
        labels: Sequence[str],
        labeled_images: bool,
    ) -> str:
        prompt = merge_system_and_question(system_prompt, question)
        try:
            model_inputs = self._build_model_inputs(prompt, images, labels, labeled_images)
            batch = _move_inputs_to_device(model_inputs, self.model_device)
            with torch.inference_mode():
                output_ids = self.model.generate(**batch, **self.generation_kwargs)
            input_ids = batch.get("input_ids")
            decoder = self.tokenizer or self.processor
            return _decode_generated(decoder, output_ids, input_ids)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("GLM generate path failed, trying chat fallback | error=%s", exc)
            return self._chat_fallback(prompt, images)


def build_transformers_adapter(config: ModelConfig, logger: logging.Logger) -> BaseTransformersAdapter:
    family = config.family
    if family in {"qwen_transformers", "qwen_vl_transformers"}:
        return QwenTransformersAdapter(config, logger)
    if family in {"glm_transformers", "glm4v_transformers"}:
        return GLMTransformersAdapter(config, logger)
    raise ValueError(f"Unsupported transformers family: {family}")

