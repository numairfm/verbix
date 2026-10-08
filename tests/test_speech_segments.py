from verbix.voice.segments import normalize_for_tts, split_chunks


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
    assert normalize_for_tts("Use *bold* and 50% & `code` → ok") == "Use bold and fifty percent and code to ok"


def test_normalizer_verbalizes_paths_component_by_component():
    assert (
        normalize_for_tts("The current working directory is /home/numair/dev/projects/verbix.")
        == "The current working directory is home, numair, dev, projects, verbix."
    )
    assert normalize_for_tts("open ~/notes/todo.md") == "open home, notes, todo dot md"


def test_normalizer_verbalizes_numbers():
    assert normalize_for_tts("All 193 tests passed") == "All one hundred ninety-three tests passed"
    assert normalize_for_tts("Pi is 3.14") == "Pi is three point one four"
    assert normalize_for_tts("in 2026") == "in two thousand twenty-six"
    assert normalize_for_tts("call_4134918 done") == "call four one three four nine one eight done"
    assert normalize_for_tts("plan v1.2.3 now") == "plan vee one point two point three now"
    assert (
        normalize_for_tts("firmware v1.2.3.4 ready")
        == "firmware vee one point two point three point four ready"
    )


def test_normalizer_reads_comma_magnitudes_as_cardinals():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("population 8,000,000") == "population eight million"
    assert normalize_for_tts("owed 1,000") == "owed one thousand"


def test_normalizer_reads_ordinals():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("finished 1st, then 2nd and 23rd") == "finished first, then second and twenty-third"


def test_normalizer_drops_numbered_bullets_before_expansion():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("1. foo") == "foo"


def test_no_glued_digit_expansion():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("2word leading") == "2word leading"
    assert normalize_for_tts("100km race") == "100km race"
    assert normalize_for_tts("3pm appointment") == "3pm appointment"


def test_star_between_digits_is_times():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("Price is 3*4 please") == "Price is three times four please"
