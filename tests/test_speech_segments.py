from verbix.voice.segments import split_chunks, normalize_for_tts


def test_eager_first_chunk_at_clause_boundary():
    assert split_chunks("Well, this is a much longer piece of spoken text here") == [
        "Well,", "this is a much longer piece of spoken text here"]


def test_no_split_inside_protected_tokens():
    assert split_chunks("Meet Dr. Smith at 3.14 for v1.2.3 release") == [
        "Meet Dr. Smith at 3.14 for v1.2.3 release"]
    assert split_chunks("see https://example.com/a.b for e.g. details here ok") == [
        "see https://example.com/a.b for e.g. details here ok"]


def test_soft_max_forces_split_at_last_clause():
    long_text = "First part here, " + "word " * 50
    chunks = split_chunks(long_text)
    assert len(chunks) >= 2
    assert all(len(c) <= 130 and len(c.split()) <= 24 for c in chunks)


def test_long_sentence_without_clauses_splits_at_word_boundaries():
    text = (
        "The history of computers spans from early mechanical calculating devices "
        "like Charles Babbage's nineteenth-century analytical engine through the vacuum "
        "tubes and transistors of the mid-twentieth century that powered massive room-sized mainframes."
    )
    chunks = split_chunks(text)
    assert len(chunks) >= 2
    assert all(len(c) <= 130 and len(c.split()) <= 24 for c in chunks)
    # Ensure words are not chopped in half (e.g. 'massive' wasn't split into 'mass' and 'ive')
    assert any("massive" in c for c in chunks)
    assert not any(c.startswith("ive ") for c in chunks)


def test_normalizer_strips_markdown_and_symbols():
    assert normalize_for_tts("Use *bold* and 50% & `code` → ok") == "Use bold and 50 percent and code to ok"
