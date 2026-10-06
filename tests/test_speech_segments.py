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
    long_text = "First part here, " + "x" * 300
    chunks = split_chunks(long_text)
    assert len(chunks) == 2 and all(len(c) <= 221 for c in chunks)


def test_normalizer_strips_markdown_and_symbols():
    assert normalize_for_tts("Use *bold* and 50% & `code` → ok") == "Use bold and 50 percent and code to ok"
