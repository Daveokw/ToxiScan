"""Text normalisation and bounded segmentation utilities."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .config import TextLimits

_BLANK_LINES = re.compile(r"\n\s*\n+")
_INLINE_WHITESPACE = re.compile(r"[^\S\n]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class ExtractedText:
    """A bounded collection of text blocks ready for classification."""

    blocks: tuple[str, ...]
    truncated: bool = False

    @property
    def character_count(self) -> int:
        return sum(len(block) for block in self.blocks)


def normalise_text(value: str) -> str:
    """Return stable readable text without invisible control characters."""

    value = unicodedata.normalize("NFKC", value.replace("\r\n", "\n").replace("\r", "\n"))
    value = "".join(
        character
        for character in value
        if character in "\n\t" or unicodedata.category(character) != "Cc"
    )
    lines = (_INLINE_WHITESPACE.sub(" ", line).strip() for line in value.splitlines())
    return "\n".join(lines).strip()


def split_paragraphs(value: str) -> Iterator[str]:
    """Yield non-empty paragraphs from untrusted extracted text."""

    cleaned = normalise_text(value)
    if not cleaned:
        return

    yield from (part.strip() for part in _BLANK_LINES.split(cleaned) if part.strip())


def _take_natural_chunk(value: str, limit: int) -> tuple[str, str]:
    candidate = value[: limit + 1]
    minimum_break = max(1, limit // 2)

    break_at = -1
    for match in _SENTENCE_BOUNDARY.finditer(candidate):
        if match.start() >= minimum_break:
            break_at = match.start()

    if break_at < minimum_break:
        break_at = candidate.rfind(" ", minimum_break, limit + 1)
    if break_at < minimum_break:
        break_at = limit

    return value[:break_at].strip(), value[break_at:].strip()


def split_long_block(value: str, limit: int) -> Iterator[str]:
    """Split a block without cutting early when a useful boundary exists."""

    remainder = value.strip()
    while len(remainder) > limit:
        chunk, remainder = _take_natural_chunk(remainder, limit)
        if chunk:
            yield chunk
    if remainder:
        yield remainder


def prepare_blocks(values: Iterable[str], limits: TextLimits) -> ExtractedText:
    """Normalise, deduplicate, segment, and cap blocks in source order."""

    accepted: list[str] = []
    seen: set[str] = set()
    character_count = 0
    truncated = False

    for value in values:
        for paragraph in split_paragraphs(value):
            for block in split_long_block(paragraph, limits.max_block_characters):
                if len(block) < limits.minimum_block_characters:
                    continue

                fingerprint = " ".join(block.casefold().split())
                if fingerprint in seen:
                    continue

                remaining = limits.max_total_characters - character_count
                if len(accepted) >= limits.max_blocks or remaining <= 0:
                    truncated = True
                    return ExtractedText(tuple(accepted), truncated)

                if len(block) > remaining:
                    shortened = block[:remaining].rstrip()
                    if len(shortened) >= limits.minimum_block_characters:
                        accepted.append(shortened)
                    truncated = True
                    return ExtractedText(tuple(accepted), truncated)

                seen.add(fingerprint)
                accepted.append(block)
                character_count += len(block)

    return ExtractedText(tuple(accepted), truncated)


def prepare_text(value: str, limits: TextLimits) -> ExtractedText:
    """Prepare one pasted or converted document as bounded text blocks."""

    return prepare_blocks((value,), limits)
