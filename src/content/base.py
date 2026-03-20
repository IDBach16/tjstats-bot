"""Abstract base class for content generators."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PostContent:
    """Payload ready to hand to poster.py."""

    text: str
    image_path: Path | None = None
    video_path: Path | None = None
    alt_text: str = ""
    tags: list[str] = field(default_factory=list)
    reply: PostContent | None = None
    replies: list[PostContent] = field(default_factory=list)


class ContentGenerator(abc.ABC):
    """Every content generator implements `generate()`."""

    name: str = "base"

    @abc.abstractmethod
    async def generate(self) -> PostContent:
        """Produce a single post's content (text + optional image)."""
        ...
