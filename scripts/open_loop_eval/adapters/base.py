"""Abstract adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class BaseTransformersAdapter(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load model-side resources."""

    @abstractmethod
    def predict(
        self,
        system_prompt: str,
        question: str,
        images: Sequence[Any],
        labels: Sequence[str],
        labeled_images: bool,
    ) -> str:
        """Run one prediction."""


class BaseVLLMAdapter(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load processor, tokenizer, and LLM runtime."""

    @abstractmethod
    def build_request(
        self,
        system_prompt: str,
        question: str,
        images: Sequence[Any],
        labels: Sequence[str],
        labeled_images: bool,
    ) -> Any:
        """Build one vLLM request."""

    @abstractmethod
    def generate_batch(self, requests: Sequence[Any]) -> list[str]:
        """Generate predictions for a batch."""

    @abstractmethod
    def generate_one(self, request: Any) -> str:
        """Generate a single prediction."""
