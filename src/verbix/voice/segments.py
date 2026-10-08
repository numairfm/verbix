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
    r"|^[ ]*\d+\.",  # line-initial ordinals (1.)
    re.IGNORECASE | re.MULTILINE,
)
_CLAUSE = re.compile(r"[,;:—]")
_SENT_END = re.compile(r"[.!?]+")
_URL = re.compile(r"https?://\S+")
_PATH = re.compile(r"(?:^|(?<=\s))(~?/(?:[\w.\-~]+/)*[\w.\-~]+)")
_VERSION = re.compile(r"(?<![\w.])([A-Za-z])(\d+(?:\.\d+)+)(?![\w.])")
_DECIMAL = re.compile(r"(?<![\w.])\d+(?:\.\d+)+\.?(?![\w.])")
_COMMA_AMOUNT = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w.])")
_ORDINAL = re.compile(r"(?<![\w.])(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_TIMES = re.compile(r"(\d)\s*\*\s*(\d)")
_INT = re.compile(r"(?<![\w.])\d[\d,]*(?!\w)")
_MD = re.compile(r"[*_`]")
_BULLET = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+")
_HASH_HEADING = re.compile(r"(?m)^\s*#{1,6}\s+")
_HASH_SHARP = re.compile(r"([A-Za-z])#")
_HASH_NUM = re.compile(r"#(\d+)")
_HASH_TAG = re.compile(r"#([A-Za-z]\w*)")
_GLOB = re.compile(r"\*\.(\w+)")
_MONEY = re.compile(r"(?<![\w$€£¥])([$€£¥])\s*(\d[\d,]*)(?:\.(\d{1,2}))?")
_WS = re.compile(r"\s+")

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
        "home" if p == "~" else p.replace(".", " dot ")
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
    t = _URL.sub("a link", text)
    t = t.replace("_", " ")
    t = _PATH.sub(_verbalize_path, t)
    # Bullets before numbers so "1. foo" drops the marker instead of
    # expanding it to "one. foo".
    t = _BULLET.sub("", t)
    t = _HASH_HEADING.sub("", t)
    t = _HASH_SHARP.sub(r"\1 sharp", t)
    t = _HASH_NUM.sub(_verbalize_hash_num, t)
    t = _HASH_TAG.sub(r"hashtag \1", t)
    t = _GLOB.sub(r"star dot \1", t)
    t = _MONEY.sub(_verbalize_money, t)
    t = _VERSION.sub(_verbalize_version, t)
    t = _COMMA_AMOUNT.sub(_verbalize_comma_amount, t)
    t = _DECIMAL.sub(_verbalize_decimal, t)
    t = _ORDINAL.sub(_verbalize_ordinal, t)
    t = _TIMES.sub(_verbalize_times, t)
    t = _INT.sub(_verbalize_int, t)
    t = _MD.sub("", t)
    t = t.replace("°F", " degrees Fahrenheit").replace("°C", " degrees C").replace("°", " degrees ")
    t = t.replace("%", " percent ").replace("&", " and ").replace("→", " to ")
    return _WS.sub(" ", t).strip()
