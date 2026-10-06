"""Clause-aware text segmentation + TTS normalization (pure; stdlib `re` only)."""

import re

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
_PATH = re.compile(r"(?:^|(?<=\s))(?:~?/|\./)(?:[\w.\-~]+/)*([\w.\-~]+)")
_BULLET = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+")
_MD = re.compile(r"[*_#`]")
_WS = re.compile(r"\s+")


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
    t = _PATH.sub(r"\1", t)
    t = _BULLET.sub("", t)
    t = _MD.sub("", t)
    t = t.replace("%", " percent").replace("&", " and ").replace("→", " to ")
    return _WS.sub(" ", t).strip()
