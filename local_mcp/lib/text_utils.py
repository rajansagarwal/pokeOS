from __future__ import annotations
import re

_UUID_RE = re.compile(r"\b[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}\b", re.I)
_HEX_TAG_RE = re.compile(r"\[[0-9a-f]{3,5}c\]bplist00.*?$", re.I)
_STREAMTYPED_RE = re.compile(r"\bstreamtyped\b", re.I)
_BRACKET_BLOCK_RE = re.compile(r"\[[^\]]+\]")

def clean_text_artifacts(s: str) -> str:
    if not s: return ""
    s = _STREAMTYPED_RE.sub("", s)
    s = _HEX_TAG_RE.sub("", s)
    s = _UUID_RE.sub("", s)
    s = re.sub(r"\b__kIM\w+\b", "", s)
    s = re.sub(r"\bNS[A-Za-z]+\b", "", s)
    s = re.sub(r"\s+", " ", s).strip(" \t\n\r-—:;,.")
    return s

def text_friendly(s: str) -> str:
    if not s: return ""
    s = _STREAMTYPED_RE.sub("", s)
    s = _BRACKET_BLOCK_RE.sub("", s)
    s = _UUID_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" \t\n\r-—:;,.")
    return s
