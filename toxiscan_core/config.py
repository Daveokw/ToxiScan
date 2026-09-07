"""Runtime limits shared by ToxiScan's ingestion and analysis layers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

MEBIBYTE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class TextLimits:
    """Bounds applied after text has been extracted from any source."""

    max_blocks: int = 250
    max_total_characters: int = 100_000
    max_block_characters: int = 6_000
    minimum_block_characters: int = 2


@dataclass(frozen=True, slots=True)
class DocumentLimits:
    """Resource limits for an uploaded document."""

    max_upload_bytes: int = 20 * MEBIBYTE
    text: TextLimits = field(default_factory=TextLimits)


@dataclass(frozen=True, slots=True)
class WebLimits:
    """Time, network, and content limits for one public web page."""

    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 12.0
    page_load_timeout_seconds: float = 20.0
    scrape_timeout_seconds: float = 30.0
    max_response_bytes: int = 5 * MEBIBYTE
    max_redirects: int = 5
    max_scrolls: int = 20
    stable_scrolls_before_stop: int = 3
    max_load_more_clicks: int = 5
    scroll_pause_seconds: float = 0.6
    text: TextLimits = field(
        default_factory=lambda: TextLimits(
            max_blocks=250,
            max_total_characters=75_000,
            max_block_characters=4_000,
        )
    )


@dataclass(frozen=True, slots=True)
class ClassifierSettings:
    """Model and batching settings for CPU-safe contextual inference."""

    model_name: str = field(
        default_factory=lambda: os.getenv(
            "TOXISCAN_MODEL_NAME", "unitary/unbiased-toxic-roberta"
        )
    )
    max_tokens: int = 512
    stride_tokens: int = 128
    batch_size: int = 8
    evidence_batch_size: int = 12
    evidence_threshold_margin: float = 0.05
    max_evidence_candidates: int = 30
    max_evidence_results: int = 5


DOCUMENT_LIMITS = DocumentLimits()
WEB_LIMITS = WebLimits()
CLASSIFIER_SETTINGS = ClassifierSettings()
