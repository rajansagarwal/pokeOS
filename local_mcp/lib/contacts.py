from __future__ import annotations
import re, pathlib
from typing import Dict, List, Optional

_PHONE_URI_PREFIX_RE = re.compile(r"^(?:tel:|sms:)", re.I)
_EXT_TAIL_RE         = re.compile(r"(?:ext\.?|x)\s*\d+\s*$", re.I)

def clean_phone_to_digits(s: Optional[str]) -> str:
    if not s: return ""
    s = s.strip()
    s = _PHONE_URI_PREFIX_RE.sub("", s)
    s = _EXT_TAIL_RE.sub("", s)
    return re.sub(r"\D", "", s)

def canonical_phone_key(s: Optional[str]) -> str:
    d = clean_phone_to_digits(s)
    return d[-10:] if len(d) >= 10 else d

def phone_match_keys(s: Optional[str]) -> List[str]:
    d = clean_phone_to_digits(s)
    if not d: return []
    keys = []
    if len(d) >= 10: keys.append(d[-10:])
    keys.append(d)
    if len(d) >= 7:  keys.append(d[-7:])
    seen, out = set(), []
    for k in keys:
        if k and k not in seen: seen.add(k); out.append(k)
    return out

def load_contacts_cache(path: str) -> tuple[Dict[str,str], Dict[str,str]]:
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Contacts cache not found: {p}")
    phone_index: Dict[str,str] = {}
    email_index: Dict[str,str] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "||" not in line: continue
            try:
                name, kind, value = line.split("||", 2)
            except ValueError:
                continue
            name = name.strip(); value = value.strip()
            if not name or not value: continue
            if kind == "phone":
                full = clean_phone_to_digits(value)
                if len(full) >= 10:
                    last10 = full[-10:]
                    phone_index.setdefault(last10, name)
                phone_index.setdefault(full, name)
                if len(full) >= 7:
                    phone_index.setdefault(full[-7:], name)
            elif kind == "email":
                email_index.setdefault(value.lower(), name)
    return phone_index, email_index

def resolve_name(handle: Optional[str],
                 phone_index: Dict[str,str],
                 email_index: Dict[str,str]) -> Optional[str]:
    if not handle: return None
    h = handle.strip()
    if "@" in h:
        return email_index.get(h.lower())
    for k in phone_match_keys(h):
        if k in phone_index:
            return phone_index[k]
    return None
