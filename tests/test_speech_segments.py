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
    assert normalize_for_tts("3D movie") == "3D movie"


def test_star_between_digits_is_times():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("Price is 3*4 please") == "Price is three times four please"


def test_sentence_final_decimal():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("Pi is 3.14.") == "Pi is three point one four."


def test_combined_comma_decimal():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("owed 1,000.50 now") == "owed one thousand point five zero now"


def test_bad_grouping_reads_digitwise():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("I owe 1,00 euros") == "I owe one zero zero euros"


def test_hashtag_trio():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("C# is great") == "C sharp is great"
    assert normalize_for_tts("Issue #42 fixed") == "Issue number forty-two fixed"
    assert normalize_for_tts("#tag trending") == "hashtag tag trending"
    assert normalize_for_tts("# Heading one") == "Heading one"


def test_glob_star():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("*.py files here") == "star dot py files here"


def test_currency_with_cents():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("costs $3.50 total") == "costs three dollars fifty total"
    assert normalize_for_tts("$1 now") == "one dollar now"
    assert normalize_for_tts("$0.50 now") == "fifty cents now"


def test_degrees_and_percent_spacing():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("5°C hot") == "five degrees C hot"
    assert normalize_for_tts("10%20% double") == "ten percent twenty percent double"


def test_time_self_verbalizes():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("meet at 10:30") == "meet at ten thirty"
    assert normalize_for_tts("meet at 3:05") == "meet at three oh five"
    assert normalize_for_tts("3pm appointment") == "three pee em appointment"


def test_dash_ladder():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("dial 9-1-1") == "dial nine one one"
    assert normalize_for_tts("open 24-7") == "open twenty-four seven"
    assert normalize_for_tts("pages 1-2") == "pages one to two"
    assert normalize_for_tts("call 555-1234") == "call five five five, one two three four"
    assert normalize_for_tts("range 1990-2000") == "range nineteen ninety to two thousand"


def test_years_and_decades():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("the year 1900") == "the year nineteen hundred"
    assert normalize_for_tts("1990s decade") == "nineteen nineties decade"
    assert normalize_for_tts("No. 5 item") == "number five item"


def test_fractions_and_operators():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("Score was 3/4 of total") == "Score was three quarters of total"
    assert normalize_for_tts("C++ and x^2 math") == "C plus plus and x squared math"
    assert normalize_for_tts("a=b equation") == "a equals b equation"
    assert normalize_for_tts("I <3 this") == "I <3 this"


def test_paths_hex_uuid_links():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("path /home/user/ here") == "path home, user here"
    assert normalize_for_tts("open /home/user/my_file.txt") == "open home, user, my file dot txt"
    assert normalize_for_tts("see [docs](https://example.com/a_b)") == "see docs"
    assert normalize_for_tts("val 0x1F here") == "val hex one ef here"
    assert normalize_for_tts("use __init__ here") == "use dunder init here"
    assert normalize_for_tts("see example.com/foo") == "see a link"


def test_splitter_never_cuts_inside_tokens():
    from verbix.voice.segments import split_chunks

    assert split_chunks("Price 8,000,000 dollars in cash today ok") == [
        "Price 8,000,000 dollars in cash today ok"
    ]
    assert split_chunks("Meeting at 10:30 in the morning sharp ok") == [
        "Meeting at 10:30 in the morning sharp ok"
    ]
    assert split_chunks("Visit No. 5 on the list today ok") == [
        "Visit No. 5 on the list today ok"
    ]
    assert split_chunks("He scored 1. She scored 2, and they both played really well today.") == [
        "He scored 1.",
        "She scored 2, and they both played really well today.",
    ]


def test_emoji_dropped_narrowly():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts("done 🎉") == "done"
    assert normalize_for_tts("café → done") == "café to done"


def test_raw_deferrals_pinned():
    from verbix.voice.segments import normalize_for_tts

    assert normalize_for_tts('He said "hello" today') == 'He said "hello" today'
    assert normalize_for_tts("Contact FBI headquarters") == "Contact FBI headquarters"
    assert normalize_for_tts("don't stop") == "don't stop"
    assert normalize_for_tts("TCP/IP is down") == "TCP/IP is down"
    assert normalize_for_tts("well-known fact") == "well-known fact"
