"""Context-preserving toxicity classification for bounded text blocks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch

from .config import CLASSIFIER_SETTINGS, ClassifierSettings

MODERATION_LABELS = (
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
)
DISPLAY_LABELS = {
    "toxicity": "Toxic",
    "severe_toxicity": "Highly Toxic",
    "obscene": "Obscene",
    "identity_attack": "Identity Attack",
    "insult": "Insult",
    "threat": "Threat",
    "sexual_explicit": "Sexually Explicit",
}

_SENTENCE_OR_LINE = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:—])\s+")


class ClassificationError(RuntimeError):
    """Raised when model inference cannot produce a trustworthy result."""


@dataclass(frozen=True, slots=True)
class ModelBundle:
    tokenizer: object
    model: object
    labels_by_id: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CategoryScore:
    key: str
    label: str
    score: float


@dataclass(frozen=True, slots=True)
class Evidence:
    text: str
    categories: tuple[CategoryScore, ...]


@dataclass(frozen=True, slots=True)
class BlockAnalysis:
    text: str
    categories: tuple[CategoryScore, ...]
    evidence: tuple[Evidence, ...] = ()

    @property
    def flagged(self) -> bool:
        return bool(self.categories)


def _ordered_labels(id2label: dict) -> tuple[str, ...]:
    try:
        return tuple(str(id2label[index]) for index in range(len(id2label)))
    except KeyError:
        try:
            return tuple(str(id2label[str(index)]) for index in range(len(id2label)))
        except KeyError as error:
            raise ClassificationError("The model has an invalid label mapping.") from error


@lru_cache(maxsize=2)
def load_model_bundle(model_name: str | None = None) -> ModelBundle:
    """Load and cache a local CPU inference bundle without remote model code."""

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    selected_model = model_name or CLASSIFIER_SETTINGS.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        selected_model,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        selected_model,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
    ).to(torch.device("cpu"))
    model.eval()
    labels_by_id = _ordered_labels(model.config.id2label)

    missing = set(MODERATION_LABELS).difference(labels_by_id)
    if missing:
        raise ClassificationError("The configured model lacks required moderation labels.")
    return ModelBundle(tokenizer, model, labels_by_id)


def _validate_threshold(value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError("Sensitivity thresholds must be between zero and one.")


def _score_texts(
    texts: Sequence[str],
    bundle: ModelBundle,
    settings: ClassifierSettings,
    *,
    batch_size: int | None = None,
) -> list[dict[str, float]]:
    if not texts:
        return []

    encoded = bundle.tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=settings.max_tokens,
        stride=settings.stride_tokens,
        return_overflowing_tokens=True,
    )
    sample_mapping = encoded.pop("overflow_to_sample_mapping")
    model_inputs = {
        name: tensor
        for name, tensor in encoded.items()
        if name in {"input_ids", "attention_mask", "token_type_ids"}
    }
    if not model_inputs:
        raise ClassificationError("The tokenizer did not produce model input tokens.")

    scores = torch.zeros((len(texts), len(bundle.labels_by_id)), dtype=torch.float32)
    window_counts = torch.zeros(len(texts), dtype=torch.int32)
    rows_per_batch = batch_size or settings.batch_size

    with torch.inference_mode():
        for start in range(0, len(sample_mapping), rows_per_batch):
            end = start + rows_per_batch
            batch = {name: tensor[start:end] for name, tensor in model_inputs.items()}
            logits = bundle.model(**batch).logits.detach().cpu()
            probabilities = torch.sigmoid(logits)

            for row_offset, row_scores in enumerate(probabilities):
                sample_index = int(sample_mapping[start + row_offset])
                scores[sample_index] = torch.maximum(scores[sample_index], row_scores)
                window_counts[sample_index] += 1

    if torch.any(window_counts == 0):
        raise ClassificationError("The tokenizer produced an incomplete analysis.")

    moderation_indices = {
        label: bundle.labels_by_id.index(label) for label in MODERATION_LABELS
    }
    return [
        {label: float(row[index]) for label, index in moderation_indices.items()}
        for row in scores
    ]


def _categories_above(
    scores: dict[str, float], threshold: float
) -> tuple[CategoryScore, ...]:
    return tuple(
        CategoryScore(key, DISPLAY_LABELS[key], scores[key])
        for key in MODERATION_LABELS
        if scores[key] >= threshold
    )


def _candidate_phrases(text: str, maximum: int) -> tuple[str, ...]:
    candidates: list[str] = []
    seen: set[str] = set()

    for sentence in _SENTENCE_OR_LINE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue

        parts = _CLAUSE_BOUNDARY.split(sentence) if len(sentence.split()) > 24 else [sentence]
        for part in parts:
            phrase = part.strip()
            if len(phrase.split()) < 3:
                continue
            fingerprint = phrase.casefold()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            candidates.append(phrase)
            if len(candidates) >= maximum:
                return tuple(candidates)

    return tuple(candidates)


def analyse_blocks(
    texts: Sequence[str],
    threshold: float = 0.5,
    *,
    evidence_threshold: float | None = None,
    settings: ClassifierSettings = CLASSIFIER_SETTINGS,
    bundle: ModelBundle | None = None,
) -> tuple[BlockAnalysis, ...]:
    """Analyse every block with overlapping windows and contextual evidence."""

    _validate_threshold(threshold)
    evidence_cutoff = threshold if evidence_threshold is None else evidence_threshold
    _validate_threshold(evidence_cutoff)

    active_bundle = bundle or load_model_bundle(settings.model_name)
    block_scores = _score_texts(texts, active_bundle, settings)
    block_categories = [
        _categories_above(scores, threshold) for scores in block_scores
    ]

    evidence_inputs: list[str] = []
    evidence_owners: list[int] = []
    for block_index, (text, categories) in enumerate(zip(texts, block_categories)):
        if not categories:
            continue
        candidates = _candidate_phrases(text, settings.max_evidence_candidates)
        evidence_inputs.extend(candidates)
        evidence_owners.extend([block_index] * len(candidates))

    evidence_by_block: dict[int, list[Evidence]] = {}
    if evidence_inputs:
        evidence_scores = _score_texts(
            evidence_inputs,
            active_bundle,
            settings,
            batch_size=settings.evidence_batch_size,
        )
        for owner, phrase, scores in zip(
            evidence_owners, evidence_inputs, evidence_scores
        ):
            parent_keys = {category.key for category in block_categories[owner]}
            categories = tuple(
                category
                for category in _categories_above(scores, evidence_cutoff)
                if category.key in parent_keys
            )
            if categories:
                evidence_by_block.setdefault(owner, []).append(
                    Evidence(phrase, categories)
                )

    analyses = []
    for index, (text, categories) in enumerate(zip(texts, block_categories)):
        evidence = evidence_by_block.get(index, [])
        evidence.sort(
            key=lambda item: max(category.score for category in item.categories),
            reverse=True,
        )
        analyses.append(
            BlockAnalysis(
                text,
                categories,
                tuple(evidence[: settings.max_evidence_results]),
            )
        )
    return tuple(analyses)
