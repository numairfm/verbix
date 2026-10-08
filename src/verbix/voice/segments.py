"""Clause-aware text segmentation + TTS normalization.

Segmentation is stdlib; normalization pairs small custom verbalizers
(paths, versions, decimals, IDs) with num2words for ordinals. Cardinals
stay hand-rolled: the pinned convention drops the "and" num2words emits
("one hundred ninety-three", not "one hundred and ninety-three").
"""

import re

from num2words import num2words

SOFT_MAX = 130
MAX_WORDS = 24
EAGER_MIN = 35

_PROTECTED = re.compile(
    r"https?://\S+"  # URLs (covers :// and dotted paths)
    r"|(?<=\d)\.(?=\d)"  # decimals / versions (3.14, v1.2.3)
    r"|\.\.\."  # ellipses
    r"|\b(?:Mr|Mrs|Ms|Dr|St|vs|etc|e\.g|i\.e)\."  # abbreviations
    r"|(?:[A-Za-z]\.){2,}"  # U.S., U.K., ...
    r"|\d{1,3}(?:,\d{3})+"  # comma magnitudes (8,000,000)
    r"|(?<=\d):(?=\d)"  # glued colon in times/scores (10:30)
    r"|No\.(?=\s*\d)"  # numbered references (No. 5)
    r"|^[ ]*\d+\.",  # line-initial ordinals (1.)
    re.IGNORECASE | re.MULTILINE,
)
_CLAUSE = re.compile(r"[,;:—]")
_SENT_END = re.compile(r"[.!?]+")
_URL = re.compile(r"https?://\S+")
_PATH = re.compile(r"(?:^|(?<=\s))(~?/(?:[\w.\-_~]+/)*[\w.\-_~]+/?)")
_MDLINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEX = re.compile(r"(?<![\w.])0[xX][0-9A-Fa-f]+(?![\w.])")
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_IP = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?![\w.])")
_MAC = re.compile(r"(?<![\w.])(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}(?![\w.])")
_VERSION = re.compile(r"(?<![\w.])([A-Za-z])(\d+(?:\.\d+)+)(?![\w.])")
_DECIMAL = re.compile(r"(?<![\w.])\d+(?:\.\d+)+\.?(?![\w.])")
_COMMA_AMOUNT = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w.])")
_ORDINAL = re.compile(r"(?<![\w.])(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_TIMES = re.compile(r"(\d+)\s*\*\s*(\d+)")
_INT = re.compile(r"(?<![\w.])\d(?:[\d,]*\d)?(?!\w)")
_MD = re.compile(r"[*_`]")
_BULLET = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+")
_HASH_HEADING = re.compile(r"(?m)^\s*#{1,6}\s+")
_HASH_SHARP = re.compile(r"([A-Za-z])#")
_HASH_NUM = re.compile(r"#(\d+)")
_HASH_TAG = re.compile(r"#([A-Za-z]\w*)")
_GLOB = re.compile(r"\*\.(\w+)")
# Narrow emoji drop: pictograph/symbol blocks + ZWJ/VS16/RI/keycap machinery.
# Deliberately excludes é — ° ± × © ® and other Latin-1/punctuation.
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D\u20E3]"
)
_FRACTION = re.compile(r"(?<![\w.])(\d+)/(\d+)(?!/\d)(?![\w.])")
_CPLUS = re.compile(r"\bC\+\+(?!\w)")
_HEART = re.compile(r"</?3\b")
_CMP = re.compile(r">=|<=|!=|==|<(?!3\b)(?!/)|(?<![=!<>])>(?!=)|(?<![=!<>])=(?!=)")
_SQUARED = re.compile(r"(\w)\^2\b")
_POWER = re.compile(r"\^")
_DUNDER = re.compile(r"__(\w+)__")
_BARE_DOMAIN = re.compile(r"(?<![\w@:/.])([\w-]+\.(?:com|org|net|io|dev|gov|edu)(?:/[\w.\-~/?#=&%]*)?)")
_MONEY = re.compile(r"(?<![\w$€£¥])([$€£¥])\s*(\d(?:[\d,]*\d)?)(?:\.(\d{1,2}))?")
_WS = re.compile(r"\s+")

_EMERGENCY = re.compile(r"\b911\b|(?<![\w.])9-1-1(?![\w.])")
_IDIOM = re.compile(r"(?<![\w.])24[/-]7(?![\w.])|(?<![\w.])50/50(?![\w.])")
_YEAR_RANGE = re.compile(r"(?<![\w.])(1\d{3}|20\d{2})-(1\d{3}|20\d{2})(?![\w.])")
_TIME_AMPM = re.compile(r"(?<![\w.])(\d{1,2}):(\d{2})(?::(\d{2}))?([APap])\.?\s*([Mm])\.?(?![\w.])")
_TIME = re.compile(r"(?<![\w.])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\w.:])")
_COLON_FALLBACK = re.compile(r"(\d+):(\d+)")
_AMPM = re.compile(r"(\d+)\s*([AaPp])\.?\s*([Mm])\.?\b")
_DASH_RUN = re.compile(r"(?<![\w.])\d+(?:-\d+)+(?![\w.])")
_NO_NUM = re.compile(r"No\.(?=\s*\d)")
_DECADE = re.compile(r"(?<![\w.])((?:1\d|20)\d)0[sS](?![\w.])")
_DATE = re.compile(r"(?<![\w.])\d+/\d+/\d+(?![\w.])")
_ARROW = re.compile(r"->|=>|<-")
_EMAIL = re.compile(r"(?<![\w@:/.])([\w.+-]+)@([\w-]+\.[\w.]+)")
_YEAR = re.compile(r"(?<![\w.])(1\d{3})(?![\w.])")

_CURRENCY_NAMES = {"$": ("dollar", "dollars"), "€": ("euro", "euros"), "£": ("pound", "pounds"),
                   "¥": ("yen", "yen")}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
          "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_DIGIT_WORDS = {str(i): w for i, w in enumerate(_ONES)}
_LETTER_NAMES = {
    "a": "ay", "b": "bee", "c": "cee", "d": "dee", "e": "ee", "f": "ef",
    "g": "gee", "h": "aych", "i": "eye", "j": "jay", "k": "kay", "l": "el",
    "m": "em", "n": "en", "o": "oh", "p": "pee", "q": "cue", "r": "ar",
    "s": "ess", "t": "tee", "u": "you", "v": "vee", "w": "doubleyou",
    "x": "ex", "y": "why", "z": "zee",
}


def _under_100(n: int) -> str:
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    return _TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else "")


def _under_1000(n: int) -> str:
    head, rest = divmod(n, 100)
    if not head:
        return _under_100(rest)
    return _ONES[head] + " hundred" + (f" {_under_100(rest)}" if rest else "")


def _int_words(n: int) -> str:
    if n < 1000:
        return _under_1000(n)
    parts = []
    for scale, name in ((10**9, "billion"), (10**6, "million"), (10**3, "thousand")):
        head, n = divmod(n, scale)
        if head:
            parts.append(f"{_under_1000(head)} {name}")
    if n:
        parts.append(_under_1000(n))
    return " ".join(parts)


def _verbalize_path(m: re.Match[str]) -> str:
    raw = m.group(0)
    trail = ""
    if raw.endswith("."):
        raw, trail = raw[:-1], "."
    parts = [
        "home" if p == "~" else p.replace("_", " ").replace(".", " dot ")
        for p in re.split(r"/+", raw)
        if p not in ("", ".")
    ]
    return ", ".join(p for p in parts if p) + trail


def _verbalize_version(m: re.Match[str]) -> str:
    letter = _LETTER_NAMES.get(m.group(1).lower(), m.group(1))
    groups = [" ".join(_DIGIT_WORDS[d] for d in g) for g in m.group(2).split(".")]
    return f"{letter} {' point '.join(groups)}"


def _verbalize_decimal(m: re.Match[str]) -> str:
    raw = m.group(0)
    trail = ""
    if raw.endswith("."):
        raw, trail = raw[:-1], "."
    chunks = raw.split(".")
    left = chunks[0]
    left_words = _int_words(int(left)) if len(left) <= 4 else " ".join(_DIGIT_WORDS[d] for d in left)
    right = " ".join(_DIGIT_WORDS[d] for d in "".join(chunks[1:]))
    return f"{left_words} point {right}{trail}"


def _verbalize_comma_amount(m: re.Match[str]) -> str:
    raw = m.group(0)
    if "." in raw:
        head, frac = raw.split(".")
        return f"{_int_words(int(head.replace(',', '')))} point {' '.join(_DIGIT_WORDS[d] for d in frac)}"
    return _int_words(int(raw.replace(",", "")))


def _verbalize_ordinal(m: re.Match[str]) -> str:
    return str(num2words(int(m.group(1)), to="ordinal"))


def _num_side(digits: str) -> str:
    return _int_words(int(digits)) if len(digits) <= 4 else " ".join(_DIGIT_WORDS[d] for d in digits)


def _verbalize_times(m: re.Match[str]) -> str:
    return f"{_num_side(m.group(1))} times {_num_side(m.group(2))}"


def _verbalize_money(m: re.Match[str]) -> str:
    singular, plural = _CURRENCY_NAMES[m.group(1)]
    amount = int(m.group(2).replace(",", ""))
    frac = m.group(3)
    cents = int(frac.ljust(2, "0")) if frac else 0
    if cents and not amount:
        return f"{_int_words(cents)} cents"
    words = f"{_int_words(amount)} {singular if amount == 1 else plural}"
    if cents:
        words += f" {_int_words(cents)}"
    return words


def _year_words(n: int) -> str:
    if n == 2000:
        return "two thousand"
    if 2001 <= n <= 2009:
        return f"two thousand {_ONES[n % 10]}"
    hi, lo = divmod(n, 100)
    return _under_100(hi) + (" hundred" if lo == 0 else f" {_under_100(lo)}")


def _digitwise(digits: str) -> str:
    return " ".join(_DIGIT_WORDS[d] for d in digits)


def _verbalize_emergency(m: re.Match[str]) -> str:
    return _digitwise(m.group(0).replace("-", ""))


def _verbalize_idiom(m: re.Match[str]) -> str:
    return {"24-7": "twenty-four seven", "24/7": "twenty-four seven"}.get(
        m.group(0), "fifty fifty"
    )


def _verbalize_year_range(m: re.Match[str]) -> str:
    return f"{_year_words(int(m.group(1)))} to {_year_words(int(m.group(2)))}"


def _minute_words(mi: str) -> str:
    if mi == "00":
        return "o'clock"
    if len(mi) == 2 and mi.startswith("0"):
        return f"oh {_ONES[int(mi[1])]}"
    return _int_words(int(mi))


def _verbalize_time(h: str, mi: str, sec: str | None, ampm: str | None) -> str:
    out = f"{_int_words(int(h))} {_minute_words(mi)}"
    if sec:
        out += f" {_minute_words(sec)}"
    if ampm:
        out += " ay em" if ampm.lower() == "a" else " pee em"
    return out


def _verbalize_time_ampm(m: re.Match[str]) -> str:
    return _verbalize_time(m.group(1), m.group(2), m.group(3), m.group(4))


def _verbalize_time_bare(m: re.Match[str]) -> str:
    return _verbalize_time(m.group(1), m.group(2), m.group(3), None)


def _verbalize_ampm(m: re.Match[str]) -> str:
    suffix = "ay em" if m.group(2).lower() == "a" else "pee em"
    return f"{_num_side(m.group(1))} {suffix}"


def _verbalize_colon_fallback(m: re.Match[str]) -> str:
    return f"{_num_side(m.group(1))} to {_num_side(m.group(2))}"


def _verbalize_dash_run(m: re.Match[str]) -> str:
    parts = m.group(0).split("-")
    if all(len(p) <= 2 for p in parts):
        return " to ".join(_num_side(p) for p in parts)
    return ", ".join(_digitwise(p) for p in parts)


def _verbalize_decade(m: re.Match[str]) -> str:
    stem = m.group(1)
    if stem == "200":
        return "two thousands"
    century = _under_100(int(stem[:2]))
    d = int(stem[2])
    if d == 0:
        return f"{century} hundreds"
    tens = "ten" if d == 1 else _TENS[d]
    return f"{century} {tens[:-1] + 'ies' if tens.endswith('y') else tens + 's'}"


def _hex_char(c: str) -> str:
    return _DIGIT_WORDS[c] if c.isdigit() else _LETTER_NAMES.get(c.lower(), c)


def _verbalize_hex(m: re.Match[str]) -> str:
    return "hex " + " ".join(_hex_char(c) for c in m.group(0)[2:])


def _verbalize_ip(m: re.Match[str]) -> str:
    return " dot ".join(" ".join(_DIGIT_WORDS[d] for d in g) for g in m.group(0).split("."))


def _verbalize_mac(m: re.Match[str]) -> str:
    return ", ".join(" ".join(_hex_char(c) for c in g) for g in m.group(0).split(":"))


def _verbalize_fraction(m: re.Match[str]) -> str:
    if m.group(0) == "9/11":
        return m.group(0)
    num, den = m.group(1), m.group(2)
    if num == "1" and den == "2":
        return "one half"
    if den == "2":
        return f"{_num_side(num)} halves"
    if den == "4":
        denom = "quarter" if num == "1" else "quarters"
        return f"{_num_side(num)} {denom}"
    return f"{_num_side(num)} {num2words(int(den), to='ordinal')}s"


def _verbalize_email(m: re.Match[str]) -> str:
    return f"{m.group(1)} at {m.group(2).replace('.', ' dot ')}"


def _verbalize_cmp(m: re.Match[str]) -> str:
    return {
        ">=": " greater than or equal to ", "<=": " less than or equal to ",
        "!=": " not equal to ", "==": " equals ", "<": " less than ",
        ">": " greater than ", "=": " equals ",
    }[m.group(0)]


def _verbalize_hash_num(m: re.Match[str]) -> str:
    digits = m.group(1)
    words = _int_words(int(digits)) if len(digits) <= 4 else " ".join(_DIGIT_WORDS[d] for d in digits)
    return f"number {words}"


def _verbalize_int(m: re.Match[str]) -> str:
    raw = m.group(0)
    digits = raw.replace(",", "")
    if "," in raw:
        # Only strict US grouping counts as a magnitude ("1,00" is not).
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", raw):
            return _int_words(int(digits))
        return " ".join(_DIGIT_WORDS[d] for d in digits)
    if len(digits) > 4:
        # IDs and long codes read digit by digit, never as magnitudes.
        return " ".join(_DIGIT_WORDS[d] for d in digits)
    return _int_words(int(digits))


def _spans(text):
    return [m.span() for m in _PROTECTED.finditer(text)]


def _boundaries(text):
    spans = _spans(text)
    cuts = set()
    for m in list(_CLAUSE.finditer(text)) + list(_SENT_END.finditer(text)):
        if not any(s <= m.start() < e for s, e in spans):
            cuts.add(m.end())
    return sorted(cuts)


def _is_valid_cut(s: str, cut: int) -> bool:
    head = s[:cut].strip()
    return len(head) <= SOFT_MAX and len(head.split()) <= MAX_WORDS


def split_chunks(text):
    s = text.strip()
    if not s:
        return []
    if len(s) >= EAGER_MIN:
        cuts = _boundaries(s)
        if cuts and _is_valid_cut(s, cuts[0]):
            head, tail = s[: cuts[0]].rstrip(), s[cuts[0]:].strip()
            if head and tail:
                return [head, *_split_tail(tail)]
    return _split_tail(s)


def _split_tail(s):
    out, cur = [], s
    while len(cur) > SOFT_MAX or len(cur.split()) > MAX_WORDS:
        punc_cuts = [c for c in _boundaries(cur) if _is_valid_cut(cur, c)]
        if punc_cuts:
            cut = punc_cuts[-1]
        else:
            word_cuts = [m.end() for m in re.finditer(r"\s+", cur) if _is_valid_cut(cur, m.end())]
            cut = word_cuts[-1] if word_cuts else SOFT_MAX
        head, cur = cur[:cut].rstrip(), cur[cut:].strip()
        if head:
            out.append(head)
    if cur:
        out.append(cur)
    return out


def normalize_for_tts(text):
    stashed: list[str] = []

    def _stash(m):
        stashed.append(m.group(0))
        return "@@STASH" + str(len(stashed) - 1) + "@@"

    t = _MDLINK.sub(r"\1", text)
    t = _URL.sub("a link", t)
    t = _PATH.sub(_verbalize_path, t)
    t = _DUNDER.sub(r"dunder \1", t)
    t = t.replace("_", " ")
    # Bullets before numbers so "1. foo" drops the marker instead of
    # expanding it to "one. foo".
    t = _BULLET.sub("", t)
    t = _HASH_HEADING.sub("", t)
    t = _HASH_SHARP.sub(r"\1 sharp", t)
    t = _HASH_NUM.sub(_verbalize_hash_num, t)
    t = _HASH_TAG.sub(r"hashtag \1", t)
    t = _GLOB.sub(r"star dot \1", t)
    t = _MONEY.sub(_verbalize_money, t)
    t = _EMERGENCY.sub(_verbalize_emergency, t)
    t = _IDIOM.sub(_verbalize_idiom, t)
    t = _YEAR_RANGE.sub(_verbalize_year_range, t)
    t = _HEX.sub(_verbalize_hex, t)
    t = _UUID.sub(_stash, t)
    t = _DATE.sub(_stash, t)
    t = _IP.sub(_verbalize_ip, t)
    t = _MAC.sub(_verbalize_mac, t)
    t = _TIME_AMPM.sub(_verbalize_time_ampm, t)
    t = _TIME.sub(_verbalize_time_bare, t)
    t = _COLON_FALLBACK.sub(_verbalize_colon_fallback, t)
    t = _AMPM.sub(_verbalize_ampm, t)
    t = _DASH_RUN.sub(_verbalize_dash_run, t)
    t = _NO_NUM.sub("number", t)
    t = _DECADE.sub(_verbalize_decade, t)
    # Pre-2000 years pair-read ("nineteen hundred"); 2000+ stays cardinal
    # per the pinned "in 2026" convention.
    t = _YEAR.sub(lambda m: _year_words(int(m.group(1))), t)
    t = _VERSION.sub(_verbalize_version, t)
    t = _COMMA_AMOUNT.sub(_verbalize_comma_amount, t)
    t = _DECIMAL.sub(_verbalize_decimal, t)
    t = _FRACTION.sub(_verbalize_fraction, t)
    t = _ORDINAL.sub(_verbalize_ordinal, t)
    t = _TIMES.sub(_verbalize_times, t)
    t = _CPLUS.sub("C plus plus", t)
    t = _HEART.sub(_stash, t)
    t = _ARROW.sub(" to ", t)
    t = _CMP.sub(_verbalize_cmp, t)
    t = _SQUARED.sub(r"\1 squared", t)
    t = _POWER.sub(" to the power of ", t)
    t = _EMAIL.sub(_verbalize_email, t)
    t = _BARE_DOMAIN.sub("a link", t)
    t = _INT.sub(_verbalize_int, t)
    t = _MD.sub("", t)
    t = _EMOJI.sub(" ", t)
    t = t.replace("°F", " degrees Fahrenheit").replace("°C", " degrees C").replace("°", " degrees ")
    t = t.replace("%", " percent ").replace("&", " and ").replace("→", " to ")
    for i, original in enumerate(stashed):
        t = t.replace("@@STASH" + str(i) + "@@", original)
    return _WS.sub(" ", t).strip()
