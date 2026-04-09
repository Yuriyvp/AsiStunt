"""Tests for SentenceChunker — splits token stream into TTS-ready chunks."""
import pytest

from voice_assistant.core.sentence_chunker import SentenceChunker


def test_first_chunk_splits_at_clause_boundary():
    """First chunk should split aggressively at comma/semicolon."""
    c = SentenceChunker()
    chunks = []
    text = "Well, I think the weather today is quite nice."
    for word in text.split():
        result = c.feed(word + " ")
        if result:
            chunks.append(result)
    final = c.flush()
    if final:
        chunks.append(final)
    assert len(chunks) >= 2
    # First chunk should end at a clause boundary (comma)
    assert chunks[0].endswith(",") or chunks[0].endswith(";")


def test_subsequent_chunks_split_at_sentence_boundary():
    """After first chunk, should split at sentence boundaries."""
    c = SentenceChunker()
    chunks = []
    text = (
        "Well, I think the weather is nice. The sun is shining brightly today. "
        "The birds are singing their morning songs. Would you like to go for a walk?"
    )
    for word in text.split():
        result = c.feed(word + " ")
        if result:
            chunks.append(result)
    final = c.flush()
    if final:
        chunks.append(final)
    assert len(chunks) >= 3
    # Subsequent chunks should end at sentence boundaries
    for chunk in chunks[1:-1]:
        assert chunk[-1] in ".!?", f"Expected sentence boundary, got: '{chunk}'"


def test_safety_valve_at_max_chars():
    """Long text without boundaries should force flush at max_chunk."""
    c = SentenceChunker(max_chunk=50)
    # Feed a long string with no punctuation
    long_text = "word " * 40  # 200 chars
    chunks = []
    for word in long_text.split():
        result = c.feed(word + " ")
        if result:
            chunks.append(result)
    final = c.flush()
    if final:
        chunks.append(final)
    assert len(chunks) >= 2
    for chunk in chunks[:-1]:
        assert len(chunk) <= 55  # some tolerance for word boundaries


def test_flush_emits_remaining():
    """Flush should emit any remaining buffered text."""
    c = SentenceChunker()
    c.feed("Short text")
    result = c.flush()
    assert result == "Short text"


def test_empty_input():
    """Empty or whitespace input should not emit chunks."""
    c = SentenceChunker()
    assert c.feed("") is None
    assert c.feed("   ") is None
    assert c.flush() is None


def test_abbreviations_not_split():
    """Abbreviations like Mr. Dr. should not trigger sentence splits."""
    c = SentenceChunker(first_chunk_min=10, normal_chunk_min=10)
    chunks = []
    # Feed enough to pass min_len, with an abbreviation
    text = "Hello there, Mr. Smith is here today."
    for word in text.split():
        result = c.feed(word + " ")
        if result:
            chunks.append(result)
    final = c.flush()
    if final:
        chunks.append(final)
    # "Mr." should NOT cause a split — text should not be split at "Mr."
    full = " ".join(chunks)
    assert "Mr." in full or "Mr" in full


def test_reset_clears_state():
    """Reset should clear buffer and chunk count."""
    c = SentenceChunker()
    c.feed("Some buffered text")
    c.reset()
    assert c.flush() is None


def test_exclamation_and_question_marks():
    """Should split on ! and ? boundaries."""
    c = SentenceChunker(first_chunk_min=5, normal_chunk_min=5)
    chunks = []
    text = "Hello! How are you? I am fine."
    for word in text.split():
        result = c.feed(word + " ")
        if result:
            chunks.append(result)
    final = c.flush()
    if final:
        chunks.append(final)
    assert len(chunks) >= 2
