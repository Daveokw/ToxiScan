from toxiscan_core.config import TextLimits
from toxiscan_core.text import normalise_text, prepare_blocks, prepare_text


def test_normalise_text_removes_controls_and_collapses_inline_spacing():
    value = "  First\t\tline\x00\r\nSecond   line  "

    assert normalise_text(value) == "First line\nSecond line"


def test_prepare_blocks_preserves_order_and_removes_exact_duplicates():
    limits = TextLimits(max_blocks=10, max_total_characters=1_000)

    result = prepare_blocks(("First block", " first   block ", "Second block"), limits)

    assert result.blocks == ("First block", "Second block")
    assert result.truncated is False


def test_prepare_text_splits_long_content_without_exceeding_block_limit():
    limits = TextLimits(max_block_characters=30, max_total_characters=1_000)

    result = prepare_text(
        "This is the first sentence. This is the second sentence. Final words.",
        limits,
    )

    assert len(result.blocks) > 1
    assert all(len(block) <= 30 for block in result.blocks)
    assert " ".join(result.blocks).replace("  ", " ").startswith("This is the first")


def test_prepare_blocks_reports_when_block_limit_is_reached():
    limits = TextLimits(max_blocks=2, max_total_characters=1_000)

    result = prepare_blocks(("One", "Two", "Three"), limits)

    assert result.blocks == ("One", "Two")
    assert result.truncated is True


def test_prepare_blocks_truncates_safely_at_character_limit():
    limits = TextLimits(
        max_blocks=10,
        max_total_characters=12,
        max_block_characters=100,
    )

    result = prepare_blocks(("First", "Second block"), limits)

    assert result.blocks == ("First", "Second")
    assert result.character_count == 11
    assert result.truncated is True


def test_prepare_text_returns_empty_result_for_blank_input():
    result = prepare_text(" \n\t ", TextLimits())

    assert result.blocks == ()
    assert result.truncated is False
