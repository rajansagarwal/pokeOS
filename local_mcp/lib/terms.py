from __future__ import annotations
import json, re
from typing import List, Set
from lib.log import debug

_TERM_STOP = {
    "the","and","but","with","from","that","this","for","you","your","me","him","her",
    "they","them","his","hers","our","ours","are","is","was","were","be","been","to",
    "of","in","on","at","as","it","if","or","so","do","does","did","will","would",
    "can","could","should","a","an","by","about","just","like","into","over","up",
    "down","out","any","some","than","then","there","here","when","where","why",
    "not","have","has","had","asked","ask","asking","draft","based","someone",
    "message","past","please","thanks","thank","hey","hi","hello"
}

def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9'\-]{2,}", text)

def _quoted_spans(text: str) -> list[str]:
    spans = []
    for a,b in re.findall(r'"([^"]{2,80})"|“([^”]{2,80})”', text):
        s = (a or b).strip()
        if s: spans.append(s)
    return spans

def _proper_spans(text: str) -> list[str]:
    return re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b", text)

def _local_ngrams_around_keywords(text: str, keep_tokens: set[str]) -> list[str]:
    toks = _tokenize_words(text)
    out = []
    L = len(toks)
    for i,t in enumerate(toks):
        tl = t.lower()
        if tl in keep_tokens:
            left = toks[i-1] if i-1 >= 0 else None
            right = toks[i+1] if i+1 < L else None
            if left and left.lower() not in _TERM_STOP:
                out.append(f"{left} {t}")
            if right and right.lower() not in _TERM_STOP:
                out.append(f"{t} {right}")
            if left and right and left.lower() not in _TERM_STOP and right.lower() not in _TERM_STOP:
                out.append(f"{left} {t} {right}")
    return out

def _seed_terms_from_task(task: str, max_terms: int = 16) -> list[str]:
    seeds = set(_quoted_spans(task)) | set(_proper_spans(task))
    for tok in _tokenize_words(task):
        tl = tok.lower()
        if tl not in _TERM_STOP:
            seeds.add(tok)
    keep_tokens = {t.lower() for t in seeds}
    for span in _local_ngrams_around_keywords(task, keep_tokens):
        seeds.add(span)

    cleaned = []
    seen = set()
    for s in seeds:
        ss = " ".join(s.split()).strip()
        if not ss: continue
        key = ss.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(ss)
    cleaned.sort(key=lambda s: (len(s.split()) > 3, len(s)))
    return cleaned[:max_terms] if cleaned else [task.strip()[:60]]

def _rank_terms(terms: list[str], llm_bonus: set[str]) -> list[str]:
    ranked = []
    for t in terms:
        L = len(t)
        words = t.split()
        proper = sum(1 for w in words if w[:1].isupper())
        has_digit = any(ch.isdigit() for ch in t)
        in_llm = t in llm_bonus or t.lower() in {x.lower() for x in llm_bonus}

        score = 0.0
        score += 1.5 if in_llm else 0.0
        score += 0.4 * proper
        score += 0.3 if has_digit else 0.0
        score += 0.2 if 3 <= len(words) <= 4 else 0.0
        score += 0.2 if 8 <= L <= 24 else 0.0
        ranked.append((score, -len(words), -L, t))
    ranked.sort(reverse=True)
    return [t for _,_,_,t in ranked]

def gen_terms_llm(task: str, model: str) -> List[str]:
    """
    Robust dynamic term extraction:
      - deterministic seeds from TASK
      - LLM 6–12 literal anchors
      - union + filter junk + rank; return top 8–12
    """
    seed_terms = _seed_terms_from_task(task, max_terms=24)
    llm_out: list[str] = []
    try:
        from openai import OpenAI
        client = OpenAI()
        sys = (
            "You extract literal search anchors for retrieval.\n"
            "Return ONLY a JSON array of 6–12 strings (no prose). Each item should be a short noun phrase or exact keyword that appears in the TASK or is an obvious literal variant (case preserved when it appears). "
            "Focus on: holiday/event names, qualifiers (e.g., Canadian/American), places, dates, people, and short action phrases already present (e.g., 'hang out'). "
            "Avoid auxiliaries and generic words (e.g., not, have, asked, draft, based, message, someone). "
            "Prefer compact, specific anchors over long sentences."
        )
        msgs = [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps({"TASK": task}, ensure_ascii=False)},
        ]
        resp = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=200, messages=msgs
        )
        txt = (resp.choices[0].message.content or "[]").strip()
        llm_out = json.loads(txt)
        llm_out = [ " ".join(s.split()) for s in llm_out if isinstance(s, str) and s.strip() ]
    except Exception:
        llm_out = []

    junk = _TERM_STOP
    merged = []
    seen = set()
    for s in (llm_out + seed_terms):
        ss = " ".join(s.split()).strip()
        if not ss: continue
        if ss.lower() in junk: continue
        if len(ss.split()) > 6: continue
        key = ss.lower()
        if key not in seen:
            seen.add(key)
            merged.append(ss)

    ranked = _rank_terms(merged, llm_bonus=set(llm_out))
    top = ranked[:12]
    if len(top) < min(8, len(ranked)):
        top = ranked[:max(8, len(top))]

    debug(f"terms={top}")
    return top
