"""Clause-aware text segmentation + TTS normalization (pure; stdlib `re` only)."""

import re

SOFT_MAX = 220
EAGER_MIN = 40

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


def split_chunks(text):
    s = text.strip()
    if not s:
        return []
    if len(s) >= EAGER_MIN:
        cuts = _boundaries(s)
        if cuts:
            head, tail = s[: cuts[0]].rstrip(), s[cuts[0]:].strip()
            if head and tail and len(head) <= SOFT_MAX and len(tail) <= SOFT_MAX:
                return [head, *_split_tail(tail)]
    return _split_tail(s)


def _split_tail(s):
    out, cur = [], s
    while len(cur) > SOFT_MAX:
        cuts = [c for c in _boundaries(cur) if c <= SOFT_MAX]
        cut = cuts[-1] if cuts else SOFT_MAX
        if len(cur) - cut > SOFT_MAX:
            cut = SOFT_MAX  # ponytail: boundary would strand oversized tail; hard cut instead
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
