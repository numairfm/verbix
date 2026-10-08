# Normalization Slice 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the deterministic TTS frontend speak every actively-wrong case correctly (symbols, glued digits, phones, currency, times, versions) with exact input→output tests.

**Architecture:** Ordered regex/map rules in `normalize_for_tts` (single pipeline, fixed order), splitter guards in `_PROTECTED`, two intent-only prompt bullets. Each task adds rules + pins; order dependencies flow forward (Task N consumes the pipeline shape Task N-1 left).

**Tech Stack:** Python ≥3.13, stdlib `re` + `num2words` (ordinals only). No new dependencies.

**Spec:** In-chat approved scope 2026-10-08 (amended Slice 1 + 2 prompt bullets; ear-approved render battery `/tmp/opencode/norm-proof.wav`). Key decisions: cardinal years kept (`2026` → `two thousand twenty-six`, no pairing churn); mid-line `1.` strip REJECTED (content loss); emoji narrow-drop only; US-only number grouping; letter-slashes stay raw; homographs/acronym stress/raw-deferrals pinned as passthrough, never regexed.

## Global Constraints

- Python >= 3.13.
- No new dependencies beyond `num2words` (already in `pyproject.toml`).
- Every rule ships with exact `input → output` test pairs in `tests/test_speech_segments.py`.
- `ruff check` clean on touched files; full suite green before each task commits.
- Displayed text stays raw; only `normalize_for_tts` output changes.
- Never broaden: no acronym stress rules, no homograph rules, no verbatim-code-fidelity work, no new prompt sections beyond the 2 approved bullets.

## Review Focus

1. Combined `1,000.50` must read `one thousand point five zero` (Task 2).
2. `He scored 1. She scored 2.` must keep both digits (Task 7 pins the rejection).
3. `café → done 🎉` must keep `café`, `→`, and drop only 🎉 (Task 8).
4. `I <3 this` must not become `less than` (Task 6).
5. `TCP/IP is down` must stay raw (Task 6 pins the refusal).

---

### Task 1: Digit-glue guard and star-times

**Files:**
- Modify: `src/verbix/voice/segments.py` (`_INT` lookarounds, new `_TIMES`, `normalize_for_tts` order)
- Test: `tests/test_speech_segments.py`

**Interfaces:**
- Consumes: existing `_INT`, `_verbalize_int`, `_MD`.
- Produces: `_TIMES` regex + `_verbalize_times` semantics (digit `*` digit → `times`) that later tasks rely on running before `_INT` and `_MD`.

- [ ] **Step 1: Write the failing tests**

```python
def test_no_glued_digit_expansion():
    assert normalize_for_tts("2word leading") == "2word leading"
    assert normalize_for_tts("100km race") == "100km race"
    assert normalize_for_tts("3pm appointment") == "3pm appointment"

def test_star_between_digits_is_times():
    assert normalize_for_tts("Price is 3*4 please") == "Price is three times four please"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider -k "glued or star"`
Expected: FAIL (glued expansion + `threefour` merge).

- [ ] **Step 3: Implement in `src/verbix/voice/segments.py`**

`_INT` gains a trailing glued-letter guard (no expansion when a letter follows the digits); new `_TIMES` (`(\d)\s*\*\s*(\d)` → `times`) runs before `_INT`; `_MD` keeps `#` removal (Task 3 narrows the class).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/segments.py tests/test_speech_segments.py
git commit -m "feat: digit-glue guard and star-times verbalization"
```

### Task 2: Decimal tolerance and comma discipline

**Files:**
- Modify: `src/verbix/voice/segments.py` (`_DECIMAL` trailing guard, combined comma-decimal rule, strict magnitude check)
- Test: `tests/test_speech_segments.py`

**Interfaces:**
- Consumes: pipeline order from Task 1.
- Produces: sentence-final decimals + `1,000.50` handled before Task 5's currency rule needs them.

- [ ] **Step 1: Write the failing tests**

```python
def test_sentence_final_decimal():
    assert normalize_for_tts("Pi is 3.14.") == "Pi is three point one four."

def test_combined_comma_decimal():
    assert normalize_for_tts("owed 1,000.50 now") == "owed one thousand point five zero now"

def test_bad_grouping_reads_digitwise():
    assert normalize_for_tts("I owe 1,00 euros") == "I owe one zero zero euros"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider -k "decimal or comma"`
Expected: FAIL (`one.50.`, `onezero`, `one hundred`).

- [ ] **Step 3: Implement in `src/verbix/voice/segments.py`**

`_DECIMAL` tolerates one sentence-final `.` (strip + re-append, same shape as `_verbalize_path`); new combined rule `\d{1,3}(?:,\d{3})+(?:\.\d+)?` before `_DECIMAL`; magnitude only on strict grouping, else digitwise.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/segments.py tests/test_speech_segments.py
git commit -m "fix: decimal tolerance and strict comma magnitudes"
```

### Task 3: Hashtag trio, glob-star, heading pins

**Files:**
- Modify: `src/verbix/voice/segments.py` (`#` rules, `_MD` drops `#`, glob rule)
- Test: `tests/test_speech_segments.py`

**Interfaces:**
- Consumes: `_MD` without `#` (later tasks assume `#` never reaches MD-strip).
- Produces: `#` fully owned; Task 9's passthrough pins assume these.

- [ ] **Step 1: Write the failing tests**

```python
def test_hashtag_trio():
    assert normalize_for_tts("C# is great") == "C sharp is great"
    assert normalize_for_tts("Issue #42 fixed") == "Issue number forty-two fixed"
    assert normalize_for_tts("#tag trending") == "hashtag tag trending"
    assert normalize_for_tts("# Heading one") == "Heading one"

def test_glob_star():
    assert normalize_for_tts("*.py files here") == "star dot py files here"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider -k "hashtag or glob"`
Expected: FAIL (`C is great`, `forty-two` without `number`, `.py files`).

- [ ] **Step 3: Implement in `src/verbix/voice/segments.py`**

Three disjoint `#` rules by next-char (letter-after-`#` preceded by letter → `sharp`; `#`+digits → `number`; `^#+\s` strip; else `#word` → `hashtag word`), all before `_MD`; remove `#` from `_MD`; glob `*.ext` → `star dot ext` before MD-strip.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/segments.py tests/test_speech_segments.py
git commit -m "feat: positional hashtag rules and glob verbalization"
```

### Task 4: Currency, degrees, percent spacing

**Files:**
- Modify: `src/verbix/voice/segments.py` (currency rule, degree map)
- Test: `tests/test_speech_segments.py`

**Interfaces:**
- Consumes: Task 2's combined comma-decimal (currency consumes `$1,000.50` whole first).
- Produces: money forms later tasks must not re-split (splitter already safe on words).

- [ ] **Step 1: Write the failing tests**

```python
def test_currency_with_cents():
    assert normalize_for_tts("costs $3.50 total") == "costs three dollars fifty total"
    assert normalize_for_tts("$1 now") == "one dollar now"
    assert normalize_for_tts("$0.50 now") == "fifty cents now"

def test_degrees_and_percent_spacing():
    assert normalize_for_tts("5°C hot") == "five degrees C hot"
    assert normalize_for_tts("10%20% double") == "ten percent twenty percent double"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider -k "currency or degrees or percent"`
Expected: FAIL (`$three`, `five°C`, `percenttwenty`).

- [ ] **Step 3: Implement in `src/verbix/voice/segments.py`**

Currency rule `(?<!\w)\$[\d,]+(\.\d{1,2})?` consuming the whole amount before decimal/INT (dollar/dollars + cents morphology, drop zero cents); `€/£/¥` named the same way; `°F/°C` → `degrees Fahrenheit/Celsius`; `%` emits with trailing space.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/segments.py tests/test_speech_segments.py
git commit -m "feat: currency, degree, and percent verbalization"
```

### Task 5: Times, colons, dashes, phones, years

**Files:**
- Modify: `src/verbix/voice/segments.py` (time, colon fallback, dash ladder, emergency/idiom, am/pm, `No.`, decade/years)
- Test: `tests/test_speech_segments.py`

**Interfaces:**
- Consumes: Tasks 1–2 (INT/decimal shapes), Task 4 (currency-first ordering kept).
- Produces: the complete number-ish pipeline Task 6's fraction rule builds on.

- [ ] **Step 1: Write the failing tests**

```python
def test_time_self_verbalizes():
    assert normalize_for_tts("meet at 10:30") == "meet at ten thirty"
    assert normalize_for_tts("meet at 3:05") == "meet at three oh five"
    assert normalize_for_tts("3pm appointment") == "three pee em appointment"

def test_dash_ladder():
    assert normalize_for_tts("dial 9-1-1") == "dial nine one one"
    assert normalize_for_tts("open 24-7") == "open twenty-four seven"
    assert normalize_for_tts("pages 1-2") == "pages one to two"
    assert normalize_for_tts("call 555-1234") == "call five five five, one two three four"
    assert normalize_for_tts("range 1990-2000") == "range nineteen ninety to two thousand"

def test_years_and_decades():
    assert normalize_for_tts("the year 1900") == "the year nineteen hundred"
    assert normalize_for_tts("1990s decade") == "nineteen nineties decade"
    assert normalize_for_tts("No. 5 item") == "number five item"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider -k "time or dash or year"`
Expected: FAIL (colon kept, magnitudes, `threepm`, `one thousand nine hundred`).

- [ ] **Step 3: Implement in `src/verbix/voice/segments.py`**

Precedence: emergency exact (`911`, `9-1-1`) → year-range (4-digit `19xx/20xx` both sides, paired words) → idioms (`24-7`, `50/50`) → short-segment `to` → phone digitwise (segment ≥3 digits or 3+ segments, groups joined `, `); time `\d{1,2}:\d{2}` self-verbalized with oh-padding + glued am/pm; bare `\d:\d` → `to`; `No.(?=\s*\d)` → `number` (capital N only); decades (`1990s`, `'90s`) + `1900`→`nineteen hundred` (cardinal years otherwise unchanged per spec).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/segments.py tests/test_speech_segments.py
git commit -m "feat: times, dash ladder, years, No. verbalization"
```

### Task 6: Fractions, code operators, paths, guards

**Files:**
- Modify: `src/verbix/voice/segments.py` (fraction, `C++`, comparators, `<3` guard, `^`, path fixes, underscore/PATH order, hex/UUID/IP/MAC guards, markdown links, dunder, bare domains)
- Test: `tests/test_speech_segments.py`

**Interfaces:**
- Consumes: full number pipeline (fraction is digit/digit only so `TCP/IP` stays raw).
- Produces: final normalize order all later checks assume.

- [ ] **Step 1: Write the failing tests**

```python
def test_fractions_and_operators():
    assert normalize_for_tts("Score was 3/4 of total") == "Score was three quarters of total"
    assert normalize_for_tts("C++ and x^2 math") == "C plus plus and x squared math"
    assert normalize_for_tts("a=b equation") == "a equals b equation"
    assert normalize_for_tts("I <3 this") == "I <3 this"

def test_paths_hex_uuid_links():
    assert normalize_for_tts("path /home/user/ here") == "path home, user here"
    assert normalize_for_tts("open /home/user/my_file.txt") == "open home, user, my file dot txt"
    assert normalize_for_tts("see [docs](https://example.com/a_b)") == "see docs"
    assert normalize_for_tts("val 0x1F here") == "val hex one ef here"
    assert normalize_for_tts("use __init__ here") == "use dunder init here"
    assert normalize_for_tts("see example.com/foo") == "see a link"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider -k "fraction or operator or path or hex or link or domain"`
Expected: FAIL.

- [ ] **Step 3: Implement in `src/verbix/voice/segments.py`**

Fraction digit/digit with half/quarter/ordinal-plural morphology + `9/11`/date/idiom guards; `C++` exact; comparators `>=<=!===<>` + `<3`/`</3` exception; `^` → `to the power of`/`squared` (`x^2` special-cased); PATH trailing-`/` + `_` filenames with URL→PATH→underscore order swap; hex/UUID/IP/MAC guards returning safe forms; markdown-link text extraction; dunder; bare `word.tld` → `a link`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/segments.py tests/test_speech_segments.py
git commit -m "feat: fractions, operators, paths, guards, links"
```

### Task 7: Splitter guards and abbreviation protection

**Files:**
- Modify: `src/verbix/voice/segments.py` (`_PROTECTED` additions)
- Test: `tests/test_speech_segments.py` (split-level tests)

**Interfaces:**
- Consumes: token shapes from Tasks 1–6 (guards must cover them).
- Produces: split correctness the render battery re-checks in Task 9.

- [ ] **Step 1: Write the failing tests**

```python
def test_splitter_never_cuts_inside_tokens():
    assert split_chunks("Price 8,000,000 dollars in cash today ok") == ["Price 8,000,000 dollars in cash today ok"]
    assert split_chunks("Meeting at 10:30 in the morning sharp ok") == ["Meeting at 10:30 in the morning sharp ok"]
    assert split_chunks("Visit No. 5 on the list today ok") == ["Visit No. 5 on the list today ok"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider -k "splitter"`
Expected: FAIL (cuts after `8,`, `10:`, `No.`).

- [ ] **Step 3: Implement in `src/verbix/voice/segments.py`**

`_PROTECTED` gains strict comma-magnitude, glued-colon time/score, and `No.(?=\s*\d)`; explicitly no mid-line `1.` guard and no `*` guard (pin the refusals with the tests above plus `He scored 1. She scored 2.` staying split).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/segments.py tests/test_speech_segments.py
git commit -m "fix: splitter guards for magnitudes, times, No."
```

### Task 8: Narrow emoji drop and raw-deferral pins

**Files:**
- Modify: `src/verbix/voice/segments.py` (emoji range rule)
- Test: `tests/test_speech_segments.py` (drop test + passthrough pins)

**Interfaces:**
- Consumes: final pipeline (emoji runs late, before WS collapse).
- Produces: nothing downstream; closes the (b)/(c) lists.

- [ ] **Step 1: Write the failing tests**

```python
def test_emoji_dropped_narrowly():
    assert normalize_for_tts("done 🎉") == "done"
    assert normalize_for_tts("café → done") == "café to done"

def test_raw_deferrals_pinned():
    assert normalize_for_tts('He said "hello" today') == 'He said "hello" today'
    assert normalize_for_tts("Contact FBI headquarters") == "Contact FBI headquarters"
    assert normalize_for_tts("don't stop") == "don't stop"
    assert normalize_for_tts("TCP/IP is down") == "TCP/IP is down"
    assert normalize_for_tts("well-known fact") == "well-known fact"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider -k "emoji or deferral"`
Expected: FAIL (codepoint passes through).

- [ ] **Step 3: Implement in `src/verbix/voice/segments.py`**

Emoji rule restricted to emoji blocks + ZWJ/regional-indicator/VS16/keycap sequences (explicitly excluding `é—°±×©®`); drop with space. No top-30 map (deferred per spec).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_speech_segments.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/segments.py tests/test_speech_segments.py
git commit -m "feat: narrow emoji drop and raw-deferral pins"
```

### Task 9: Prompt bullets, full verification, render battery

**Files:**
- Modify: `src/verbix/minisweagent/config/worker.yaml` (no — worker untouched), `src/verbix/minisweagent/config/voice.yaml`, `src/verbix/runtime/runtime.py` (`DEFAULT_ANNOUNCER_PROMPT`, fallback text)
- Test: full suite + render battery re-run

**Interfaces:**
- Consumes: all rules (prompt must not duplicate spellings).
- Produces: the shippable slice.

- [ ] **Step 1: Append the 2 approved intent-only bullets**

Foreground `voice.yaml` voice-constraints list + mirrored announcer text in `runtime.py`: (1) prefer plain words over symbols, versions, long numbers — reword or briefly describe rather than reciting; (2) never recite links, paths, or code character-by-character — summarize or drop. No spelling examples, no new headings, no worker changes.

- [ ] **Step 2: Run the full suite and lint**

Run: `.venv/bin/python -m pytest -q` (expected: all pass) and `.venv/bin/ruff check` on touched files (expected: clean; pre-existing baseline findings untouched).

- [ ] **Step 3: Re-run the render battery**

Synthesize the 20 approved spoken forms through `PocketHelperEngine`, confirm non-empty audio with sane durations. Re-render `/tmp/opencode/norm-proof.wav` only if outputs changed.

- [ ] **Step 4: Commit**

```bash
git add src/verbix/minisweagent/config/voice.yaml src/verbix/runtime/runtime.py
git commit -m "feat: speakable-phrasing bullets for foreground and announcer"
```

## Self-Review

1. **Spec coverage:** every (a)-severity row maps: glue/times→T1, decimal/comma→T2, `#`/glob→T3, currency/degree/percent→T4, time/dash/phone/years/No.→T5, fraction/operators/paths/guards/links→T6, splitter→T7, emoji/deferrals→T8, prompt→T9. (b)-rows: emails/mentions→T6 link/email handling (bare domains; full `bob at example dot com` — T6's link step covers domains; add email pair to T6 tests at implementation if missing: `bob@example.com` → `bob at example dot com`), `->`/`=>`→T6, `a|b`/`~`/`^`→T6 comparator/symbol maps, dunder→T6, bare domains→T6. Decades→T5. US-only/European notes→T2 tests. Cardinal years kept (no task churns them).
2. **Step scan:** each test step carries exact assertions; each code step names exact regexes/functions/files; verification steps name commands + expected output. Fixed inline: T6 email pair noted above.
3. **Type consistency:** function names (`_verbalize_*`, `_TIMES`, `normalize_for_tts`) match existing module conventions; test file and assertions reference real signatures.
4. **Review Focus:** the five lines each name an input + expected behavior, and each maps to its owning task's tests (T2, T7-rejection, T8, T6, T6).
5. **Proportion:** plan states decisions + tests; bodies left to implementer. No transcripts beyond short witness pairs.
