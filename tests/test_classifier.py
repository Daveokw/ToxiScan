from types import SimpleNamespace

import pytest
import torch

from toxiscan_core.classifier import (
    MODERATION_LABELS,
    ModelBundle,
    _normalise_for_inference,
    _score_texts,
    analyse_blocks,
)
from toxiscan_core.config import ClassifierSettings

NEUTRAL_UPPERCASE_HEADER = (
    "LADOKE AKINTOLA UNIVERSITY OF TECHNOLOGY P.M.B 4000, "
    "OGBOMOSO, OYO STATE, NIGERIA OFFICE OF THE REGISTRAR"
)


class StubTokenizer:
    def __call__(self, texts, **kwargs):
        codes = []
        owners = []
        for owner, text in enumerate(texts):
            if text == "long":
                text_codes = (0, 2)
            elif "awful" in text.casefold():
                text_codes = (1,)
            elif text == "everything":
                text_codes = (3,)
            elif text == "A neutral sentence. Another neutral sentence.":
                text_codes = (4,)
            elif text == NEUTRAL_UPPERCASE_HEADER:
                text_codes = (5,)
            else:
                text_codes = (0,)
            codes.extend(text_codes)
            owners.extend([owner] * len(text_codes))

        return {
            "input_ids": torch.tensor([[code] for code in codes]),
            "attention_mask": torch.ones((len(codes), 1), dtype=torch.int64),
            "overflow_to_sample_mapping": torch.tensor(owners),
        }


class StubModel:
    def __init__(self, label_count: int) -> None:
        self.label_count = label_count
        self.batch_sizes: list[int] = []

    def __call__(self, input_ids, attention_mask=None):
        self.batch_sizes.append(len(input_ids))
        logits = torch.full((len(input_ids), self.label_count), -10.0)
        for row, code in enumerate(input_ids[:, 0].tolist()):
            if code == 1:
                logits[row, 0] = 4.0
                logits[row, 4] = 3.0
            elif code == 2:
                logits[row, 5] = 5.0
            elif code == 3:
                logits[row, :] = 5.0
            elif code == 4:
                logits[row, 0] = 4.0
            elif code == 5:
                logits[row, 0] = 4.0
                logits[row, 3] = 3.0
        return SimpleNamespace(logits=logits)


def make_bundle(labels=MODERATION_LABELS):
    return ModelBundle(StubTokenizer(), StubModel(len(labels)), tuple(labels))


def test_score_texts_aggregates_maximum_across_overlapping_windows():
    bundle = make_bundle()
    settings = ClassifierSettings(batch_size=1)

    results = _score_texts(("long", "safe"), bundle, settings)

    assert results[0]["threat"] > 0.99
    assert results[1]["threat"] < 0.01
    assert bundle.model.batch_sizes == [1, 1, 1]


def test_score_texts_excludes_identity_attribute_outputs():
    labels = (*MODERATION_LABELS, "male")
    bundle = make_bundle(labels)

    result = _score_texts(("everything",), bundle, ClassifierSettings())[0]

    assert set(result) == set(MODERATION_LABELS)
    assert "male" not in result


def test_analyse_blocks_returns_contextual_sentence_evidence():
    bundle = make_bundle()

    result = analyse_blocks(
        ("You are awful. Thanks for helping.",),
        threshold=0.5,
        bundle=bundle,
    )[0]

    assert [category.key for category in result.categories] == ["toxicity", "insult"]
    assert [evidence.text for evidence in result.evidence] == ["You are awful."]
    assert result.flagged is True


def test_analyse_blocks_keeps_unflagged_blocks_in_source_order():
    bundle = make_bundle()

    results = analyse_blocks(("safe", "You are awful."), bundle=bundle)

    assert [result.text for result in results] == ["safe", "You are awful."]
    assert results[0].flagged is False
    assert results[1].flagged is True


def test_uppercase_document_header_is_normalised_before_inference():
    bundle = make_bundle()
    raw_scores = _score_texts(
        (NEUTRAL_UPPERCASE_HEADER,), bundle, ClassifierSettings()
    )[0]
    result = analyse_blocks(
        (NEUTRAL_UPPERCASE_HEADER,), threshold=0.5, bundle=bundle
    )[0]

    assert raw_scores["toxicity"] > 0.9
    assert _normalise_for_inference(NEUTRAL_UPPERCASE_HEADER) == (
        NEUTRAL_UPPERCASE_HEADER.casefold()
    )
    assert result.flagged is False


def test_unconfirmed_contextual_category_is_not_returned():
    result = analyse_blocks(
        ("A neutral sentence. Another neutral sentence.",),
        threshold=0.5,
        bundle=make_bundle(),
    )[0]

    assert result.flagged is False
    assert result.categories == ()
    assert result.evidence == ()


@pytest.mark.parametrize("threshold", (-0.1, 1.1))
def test_analyse_blocks_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="between zero and one"):
        analyse_blocks(("text",), threshold=threshold, bundle=make_bundle())


def test_analyse_blocks_returns_empty_tuple_for_empty_input():
    assert analyse_blocks((), bundle=make_bundle()) == ()
