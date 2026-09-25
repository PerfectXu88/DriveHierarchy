"""vLLM-backed model adapters."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import torch

from ..config import ModelConfig
from ..prompting import content_with_images, merge_system_and_question
from .base import BaseVLLMAdapter

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # noqa: BLE001
    process_vision_info = None


def _import_transformers() -> Any:
    import transformers

    return transformers


def _import_vllm() -> tuple[Any, Any]:
    from vllm import LLM, SamplingParams

    return LLM, SamplingParams


def _pick_dtype(config: ModelConfig) -> str:
    if config.runtime.dtype:
        return config.runtime.dtype
    if torch.cuda.is_available() and config.runtime.prefer_bf16 and torch.cuda.is_bf16_supported():
        return "bfloat16"
    if torch.cuda.is_available():
        return "float16"
    return "auto"


def _build_sampling_params(config: ModelConfig, stop_token_ids: Optional[list[int]] = None) -> Any:
    _, SamplingParams = _import_vllm()
    kwargs = {
        "temperature": config.generation.temperature,
        "max_tokens": config.generation.max_new_tokens,
        "top_p": config.generation.top_p,
        "top_k": config.generation.top_k,
        "repetition_penalty": config.generation.repetition_penalty,
        "presence_penalty": config.generation.presence_penalty,
    }
    if stop_token_ids:
        kwargs["stop_token_ids"] = stop_token_ids
    return SamplingParams(**kwargs)


def _extract_text_from_outputs(outputs: Any) -> list[str]:
    return [item.outputs[0].text.strip() for item in outputs]


class BaseLoadedVLLMAdapter(BaseVLLMAdapter):
    def __init__(self, config: ModelConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.processor: Any = None
        self.tokenizer: Any = None
        self.llm: Any = None
        self.sampling_params: Any = None

    def _load_processor(self) -> None:
        transformers = _import_transformers()
        kwargs: dict[str, Any] = {"trust_remote_code": self.config.runtime.trust_remote_code}
        self.processor = transformers.AutoProcessor.from_pretrained(self.config.model, **kwargs)
        self.tokenizer = getattr(self.processor, "tokenizer", None)

    def _build_llm(self) -> None:
        LLM, _ = _import_vllm()
        runtime = self.config.runtime
        llm_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "trust_remote_code": runtime.trust_remote_code,
            "dtype": _pick_dtype(self.config),
            "gpu_memory_utilization": runtime.gpu_memory_utilization,
        }
        if runtime.max_model_len:
            llm_kwargs["max_model_len"] = runtime.max_model_len
        if runtime.tensor_parallel_size:
            llm_kwargs["tensor_parallel_size"] = runtime.tensor_parallel_size
        if runtime.max_images_per_prompt:
            llm_kwargs["limit_mm_per_prompt"] = {"image": runtime.max_images_per_prompt}
        if runtime.enable_mm_preprocessor_cache:
            llm_kwargs["disable_mm_preprocessor_cache"] = False
            llm_kwargs["mm_processor_cache_gb"] = runtime.mm_processor_cache_gb
            llm_kwargs["mm_processor_cache_type"] = runtime.mm_processor_cache_type
            llm_kwargs["mm_shm_cache_max_object_size_mb"] = runtime.mm_shm_cache_max_object_size_mb
        self.llm = LLM(**llm_kwargs)

    def generate_batch(self, requests: Sequence[Any]) -> list[str]:
        outputs = self.llm.generate(list(requests), sampling_params=self.sampling_params)
        return _extract_text_from_outputs(outputs)

    def generate_one(self, request: Any) -> str:
        outputs = self.llm.generate([request], sampling_params=self.sampling_params)
        return _extract_text_from_outputs(outputs)[0]


class GenericVLLMAdapter(BaseLoadedVLLMAdapter):
    def load(self) -> None:
        try:
            self._load_processor()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Processor load failed, will rely on manual prompt fallback | error=%s", exc)
            self.processor = None
            self.tokenizer = None
        self._build_llm()
        self.sampling_params = _build_sampling_params(self.config)

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raise RuntimeError("No chat template available")

    def _build_messages(
        self,
        system_prompt: str,
        question: str,
        images: Sequence[Any],
        labels: Sequence[str],
        labeled_images: bool,
    ) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_with_images(question, images, labels, labeled_images)},
        ]

    def _manual_prompt(self, system_prompt: str, question: str, image_count: int) -> str:
        merged = merge_system_and_question(system_prompt, question)
        manual_prompt = str(self.config.family_options.get("manual_prompt", "generic")).strip().lower()
        if manual_prompt == "glm4v":
            image_tokens = "".join("<|begin_of_image|><|image|><|end_of_image|>\n" for _ in range(image_count))
            return (
                "[gMASK]<sop><|system|>\nYou are a helpful assistant."
                "<|user|>\n"
                f"{image_tokens}{merged}"
                "<|assistant|>assistant\n"
            )
        if manual_prompt == "gemma3":
            return f"<bos><start_of_turn>user\n{'<start_of_image>' * image_count}{merged}<end_of_turn>\n<start_of_turn>model\n"
        if manual_prompt == "llava15":
            image_tokens = "\n".join("<image>" for _ in range(image_count))
            return f"USER: {image_tokens}\n{merged}\nASSISTANT:" if image_tokens else f"USER: {merged}\nASSISTANT:"
        if manual_prompt == "kimi_vl":
            image_tokens = "".join(
                "<|media_start|>image<|media_content|><|media_pad|><|media_end|>" for _ in range(image_count)
            )
            return f"<|im_user|>user<|im_middle|>{image_tokens}{merged}<|im_end|><|im_assistant|>assistant<|im_middle|>"
        return merged

    def build_request(
        self,
        system_prompt: str,
        question: str,
        images: Sequence[Any],
        labels: Sequence[str],
        labeled_images: bool,
    ) -> Any:
        mm_data = {"image": images[0]} if len(images) == 1 else {"image": list(images)}
        try:
            prompt = self._apply_chat_template(self._build_messages(system_prompt, question, images, labels, labeled_images))
        except Exception:
            prompt = self._manual_prompt(system_prompt, question, len(images))
        return {"prompt": prompt, "multi_modal_data": mm_data}


class QwenVLLMAdapter(BaseLoadedVLLMAdapter):
    def load(self) -> None:
        self._load_processor()
        self._build_llm()
        self.sampling_params = _build_sampling_params(self.config)

    def build_request(
        self,
        system_prompt: str,
        question: str,
        images: Sequence[Any],
        labels: Sequence[str],
        labeled_images: bool,
    ) -> Any:
        if self.processor is None:
            raise RuntimeError("Qwen adapter requires processor")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_with_images(question, images, labels, labeled_images)},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if process_vision_info is None:
            return {
                "prompt": prompt,
                "multi_modal_data": {"image": images[0]} if len(images) == 1 else {"image": list(images)},
            }

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.config.runtime.image_patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data: dict[str, Any] = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        request = {"prompt": prompt, "multi_modal_data": mm_data}
        if video_kwargs:
            request["mm_processor_kwargs"] = video_kwargs
        return request


class InternVLVLLMAdapter(BaseLoadedVLLMAdapter):
    STOP_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]

    def load(self) -> None:
        self._load_processor()
        if self.tokenizer is None:
            transformers = _import_transformers()
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=self.config.runtime.trust_remote_code,
            )
        self._build_llm()

        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, "get_vocab") else {}
        stop_token_ids = [int(vocab[token]) for token in self.STOP_TOKENS if token in vocab]
        self.sampling_params = _build_sampling_params(self.config, stop_token_ids=stop_token_ids)

    def build_request(
        self,
        system_prompt: str,
        question: str,
        images: Sequence[Any],
        labels: Sequence[str],
        labeled_images: bool,
    ) -> Any:
        placeholder = str(self.config.family_options.get("image_placeholder", "<image>"))
        placeholders = "".join(f"Image-{index}: {placeholder}\n" for index in range(1, len(images) + 1))
        body = question.strip()
        user_content = f"{placeholders}{body}" if body else placeholders
        messages: list[dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_content})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "multi_modal_data": {"image": list(images)}}


class MiniCPMVLLMAdapter(BaseLoadedVLLMAdapter):
    IMAGE_PLACEHOLDER = "(<image>./</image>)"
    STOP_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]

    def load(self) -> None:
        self._load_processor()
        if self.tokenizer is None:
            transformers = _import_transformers()
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.model,
                trust_remote_code=self.config.runtime.trust_remote_code,
            )
        self._build_llm()
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, "get_vocab") else {}
        stop_token_ids = [int(vocab[token]) for token in self.STOP_TOKENS if token in vocab]
        self.sampling_params = _build_sampling_params(self.config, stop_token_ids=stop_token_ids)

    def build_request(
        self,
        system_prompt: str,
        question: str,
        images: Sequence[Any],
        labels: Sequence[str],
        labeled_images: bool,
    ) -> Any:
        parts: list[str] = []
        if labeled_images and labels:
            for label in labels[: len(images)]:
                parts.append(f"Image {label}:")
                parts.append(self.IMAGE_PLACEHOLDER)
        else:
            parts.extend(self.IMAGE_PLACEHOLDER for _ in images)
        if question.strip():
            parts.append(question.strip())
        user_content = "\n".join(parts)

        messages: list[dict[str, str]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_content})

        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        enable_thinking = bool(self.config.family_options.get("enable_thinking", False))
        try:
            prompt = self.tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, **kwargs)
        return {"prompt": prompt, "multi_modal_data": {"image": list(images)}}


class PixtralVLLMAdapter(BaseLoadedVLLMAdapter):
    IMAGE_PLACEHOLDER = "[IMG]"

    def load(self) -> None:
        self._load_processor()
        self._build_llm()
        self.sampling_params = _build_sampling_params(self.config)

    def _build_messages(
        self,
        system_prompt: str,
        question: str,
        image_count: int,
        labels: Sequence[str],
        labeled_images: bool,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if labeled_images and labels:
            for label in labels[:image_count]:
                content.append({"type": "text", "content": f"Image {label}:"})
                content.append({"type": "image"})
        else:
            for _ in range(image_count):
                content.append({"type": "image"})
        content.append({"type": "text", "content": question})
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": content})
        return messages

    def build_request(
        self,
        system_prompt: str,
        question: str,
        images: Sequence[Any],
        labels: Sequence[str],
        labeled_images: bool,
    ) -> Any:
        try:
            prompt = self.processor.apply_chat_template(
                self._build_messages(system_prompt, question, len(images), labels, labeled_images),
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            merged = merge_system_and_question(system_prompt, question)
            lines: list[str] = []
            if labeled_images and labels:
                for label in labels[: len(images)]:
                    lines.append(f"Image {label}: {self.IMAGE_PLACEHOLDER}")
            else:
                lines.extend(self.IMAGE_PLACEHOLDER for _ in images)
            if merged:
                lines.append(merged)
            prompt = f"<s>[INST]{chr(10).join(lines)}[/INST]"
        return {"prompt": prompt, "multi_modal_data": {"image": list(images)}}


def build_vllm_adapter(config: ModelConfig, logger: logging.Logger) -> BaseVLLMAdapter:
    family = config.family
    if family == "qwen_vl":
        return QwenVLLMAdapter(config, logger)
    if family == "generic_vllm":
        return GenericVLLMAdapter(config, logger)
    if family == "internvl":
        return InternVLVLLMAdapter(config, logger)
    if family == "minicpm":
        return MiniCPMVLLMAdapter(config, logger)
    if family == "pixtral":
        return PixtralVLLMAdapter(config, logger)
    raise ValueError(f"Unsupported vLLM family: {family}")
