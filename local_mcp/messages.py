#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse
from dotenv import load_dotenv
from typing import Any, Dict, List

# Bootstrap env
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("Set OPENAI_API_KEY in .env")

# Local imports
from lib.config import (
    DEFAULT_CONTACTS,
    DEFAULT_LANCEDB_DIR,
    DEFAULT_LANCEDB_TABLE,
    RECENT_CHATS,
    MESSAGES_PER_CHAT,
)
from lib.prompts import SUGGEST_SYSTEM_PROMPT_V2
from lib.prompt_utils import is_question
from lib.text_utils import text_friendly
from lib.retrieval import gather_topic_evidence
from models.memory import get_store, LanceMemory
from models.imessage_db import (
    read_recent_conversations_json,
    read_recent_conversations_for_indexing,
)

def suggest_next_message_from_prompt(
    user_prompt: str,
    *,
    model: str = "gpt-4o-mini",
    max_tokens: int = 220,
) -> dict:
    """
    Compose either:
      - a direct answer to the prompter (if user_prompt is a question), with a short 'reason' appended, or
      - a pasteable reply to send (if user_prompt is an ask to draft).
    Returns JSON with reply, rationale, verdict, used, and context.evidence_json.
    """
    from openai import OpenAI
    client = OpenAI()

    store = get_store()  # cached LanceMemory

    # 1) Gather compact, cleaned evidence
    evidence_items = gather_topic_evidence(store, user_prompt, retriever_model=model)

    # 2) Pack evidence payload (keep both raw & clean text; composer uses clean_text)
    evidence_json = [
        {
            "chat": r.get("chat",""),
            "sender": r.get("sender",""),
            "sender_is_me": bool(r.get("sender_is_me", False)),
            "text": (r.get("text","") or "")[:500],
            "clean_text": (r.get("clean_text","") or "")[:500],
            "timestamp": r.get("timestamp",""),
        }
        for r in evidence_items
    ]

    # 3) Compose with strict system prompt
    messages = [
        {"role": "system", "content": SUGGEST_SYSTEM_PROMPT_V2},
        {"role": "user", "content": json.dumps({"TASK": user_prompt, "EVIDENCE_JSON": evidence_json}, ensure_ascii=False)},
    ]
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        messages=messages,
    )
    raw = (resp.choices[0].message.content or "{}").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"verdict":"unknown","reply":"","rationale":"Could not parse model output."}

    # 4) Final, chat-friendly reply + optional attached rationale for Q&A mode
    reply = text_friendly(parsed.get("reply",""))
    rationale = text_friendly(parsed.get("rationale",""))
    prompter_answer = is_question(user_prompt)

    if prompter_answer and rationale:
        sep = " — " if not reply.endswith((".", "!", "?")) else " "
        reply = f"{reply}{sep}reason: {rationale}"

    candidate_chats = list({e["chat"] for e in evidence_json if e.get("chat")})[:2]
    return {
        "reply": reply,
        "rationale": rationale,
        "verdict": parsed.get("verdict","unknown"),
        "used": {
            "model": model,
            "max_tokens": max_tokens,
            "reply_mode": "prompter_answer" if prompter_answer else "pasteable_reply",
            "candidate_chats": candidate_chats,
            "evidence_count": len(evidence_json),
            "dynamic_terms": True,
        },
        "context": { "evidence_json": evidence_json }
    }

# ----------------- CLI -----------------
def main() -> None:
    """
    Interact with the local chat memory.
    """

    ap = argparse.ArgumentParser(
        description="Read recent Messages; index & query with LanceDB."
    )
    ap.add_argument("--contacts", default=DEFAULT_CONTACTS, help="Path to contacts cache file")
    ap.add_argument("--chats", type=int, default=RECENT_CHATS, help="How many recent chats")
    ap.add_argument("--per", type=int, default=MESSAGES_PER_CHAT, help="How many messages per chat")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    # LanceDB location
    ap.add_argument("--dbdir", default=DEFAULT_LANCEDB_DIR, help="LanceDB directory (default: ~/.chat_memdb)")
    ap.add_argument("--table", default=DEFAULT_LANCEDB_TABLE, help="LanceDB table name (default: messages)")

    # Indexing action (can be combined with exactly one read mode below)
    ap.add_argument("--index", action="store_true", help="Index recent messages into LanceDB")

    # Mutually exclusive read modes
    mode = ap.add_mutually_exclusive_group(required=False)
    mode.add_argument("--query", help="Semantic query to run against LanceDB")
    mode.add_argument("--text", help="Plain substring search (iMessage-style)")
    mode.add_argument("--context", help="Pull relevant context (grouped by chat) for this prompt")

    # Shared tuning flags
    ap.add_argument("--k", type=int, default=3, help="Top-K results for --query")
    ap.add_argument("--limit", type=int, default=20, help="Max results for --text search")
    ap.add_argument("--ctx_k", type=int, default=5, help="Messages per thread for --context")
    ap.add_argument("--ctx_threads", type=int, default=2, help="Max threads to return for --context")

    # Optional filters for --text
    ap.add_argument("--inchat", help="Restrict text search to a specific chat name")
    ap.add_argument("--from", dest="from_sender", help="Restrict to sender display name (e.g., 'You' or 'Rajan')")
    ap.add_argument("--since", help="ISO lower bound timestamp/date (e.g. 2025-09-10)")
    ap.add_argument("--until", help="ISO upper bound timestamp/date (e.g. 2025-09-14)")
    ap.add_argument("--window", type=int, default=0, help="Include N messages of context before/after each hit")

    # Optional: run one-shot suggestion from a prompt
    ap.add_argument("--suggest", help="Run suggest_next_message_from_prompt(TASK) and print JSON")

    args = ap.parse_args()

    # Default: dump recent conversations as JSON (no LanceDB needed)
    if not any([args.index, args.query, args.text, args.context, args.suggest]):
        data = read_recent_conversations_json(args.contacts, args.chats, args.per)
        print(json.dumps(data, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    # Open LanceDB store
    store: LanceMemory = get_store(db_dir=args.dbdir, table_name=args.table)

    # Optional indexing pass first
    if args.index:
        rows = read_recent_conversations_for_indexing(args.contacts, args.chats, args.per)
        n = store.upsert(rows, skip_existing=True)
        print(json.dumps({"indexed": n, "dbdir": args.dbdir, "table": args.table}, indent=2 if args.pretty else None))

    # Exactly one read mode
    if args.query:
        hits = store.search(args.query, k=args.k)
        print(json.dumps(hits, ensure_ascii=False, indent=2 if args.pretty else None))

    elif args.text:
        out = store.text_search(
            query=args.text,
            limit=args.limit,
            chat=args.inchat,
            sender=args.from_sender,
            since=args.since,
            until=args.until,
            window=args.window,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2 if args.pretty else None))

    elif args.context:
        ctx = store.relevant_context(args.context, k_per_thread=args.ctx_k, max_threads=args.ctx_threads)
        print(json.dumps(ctx, ensure_ascii=False, indent=2 if args.pretty else None))

    elif args.suggest:
        result = suggest_next_message_from_prompt(args.suggest)
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))

# ===== Delete the old TOPIC_CANON_TERMS and _extract_topic_terms =====
# (remove the whole TOPIC_CANON_TERMS dict and any function that depended on it)

import json, re
from typing import List, Dict, Any, Set, Optional

# --- display & evidence cleaning helpers ---

_UUID_RE = re.compile(r"\b[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}\b", re.I)
_HEX_TAG_RE = re.compile(r"\[[0-9a-f]{3,5}c\]bplist00.*?$", re.I)
_STREAMTYPED_RE = re.compile(r"\bstreamtyped\b", re.I)
_BRACKET_BLOCK_RE = re.compile(r"\[[^\]]+\]")  # used for reply-friendly “reason” only

def _clean_text_artifacts(s: str) -> str:
    """Remove iMessage parsing junk so the model sees clean content."""
    if not s:
        return ""
    s = _STREAMTYPED_RE.sub("", s)
    s = _HEX_TAG_RE.sub("", s)
    s = _UUID_RE.sub("", s)
    # remove stray Apple rich-text scaffolding leftovers like __kIM..., NS..., etc.
    s = re.sub(r"\b__kIM\w+\b", "", s)
    s = re.sub(r"\bNS[A-Za-z]+\b", "", s)
    # collapse whitespace and trim
    s = re.sub(r"\s+", " ", s).strip(" \t\n\r-—:;,.")
    return s

def _text_friendly(s: str) -> str:
    """No timestamps/citations/streamtyped in user-facing text."""
    if not s:
        return ""
    s = _STREAMTYPED_RE.sub("", s)
    s = _BRACKET_BLOCK_RE.sub("", s)  # strip things like [Chat — Sender]
    s = _UUID_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" \t\n\r-—:;,.")
    return s

def _is_question(task: str) -> bool:
    t = task.strip()
    if "?" in t:
        return True
    return bool(re.match(r"^(?i)(is|are|am|do|does|did|can|could|should|will|would|when|where|who|whom|whose|which|what|why|how)\b", t))


# --- improved salient term helpers (drop-in) ---

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
    # Grab Proper Noun sequences like "New York", "Canadian Thanksgiving"
    return re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b", text)

def _local_ngrams_around_keywords(text: str, keep_tokens: set[str]) -> list[str]:
    toks = _tokenize_words(text)
    out = []
    L = len(toks)
    for i,t in enumerate(toks):
        tl = t.lower()
        if tl in keep_tokens:
            # expand to nearby bigrams/trigrams if neighbors aren't stopwords
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
    # 1) quoted phrases and proper-noun spans
    seeds = set(_quoted_spans(task)) | set(_proper_spans(task))

    # 2) non-stopword tokens (>=3 chars)
    for tok in _tokenize_words(task):
        tl = tok.lower()
        if tl not in _TERM_STOP:
            seeds.add(tok)

    # 3) ensure we consider local phrases around obviously-meaningful tokens
    keep_tokens = {t.lower() for t in seeds}
    for span in _local_ngrams_around_keywords(task, keep_tokens):
        seeds.add(span)

    # sanitize / dedupe
    cleaned = []
    seen = set()
    for s in seeds:
        ss = " ".join(s.split()).strip()
        if not ss: 
            continue
        key = ss.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(ss)

    # prefer compact, concrete strings first
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

        # scoring: reward LLM hits, proper-noun content, compactness
        score = 0.0
        score += 1.5 if in_llm else 0.0
        score += 0.4 * proper
        score += 0.3 if has_digit else 0.0
        score += 0.2 if 3 <= len(words) <= 4 else 0.0
        score += 0.2 if 8 <= L <= 24 else 0.0
        ranked.append((score, -len(words), -L, t))

    ranked.sort(reverse=True)
    return [t for _,_,_,t in ranked]

def _gen_terms_llm(task: str, model: str) -> List[str]:
    """
    Robust dynamic term extraction:
      - Get deterministic seeds from the TASK (quoted/proper spans, non-stopwords, local n-grams).
      - Ask the LLM for 6–12 literal anchors (names, qualifiers, dates, places).
      - Union + filter junk + rank; return the top 8–12.
    """
    # 1) deterministic seeds so obvious anchors like "Thanksgiving" never drop
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

    # 2) union + filter junk
    junk = _TERM_STOP
    merged = []
    seen = set()
    for s in (llm_out + seed_terms):
        ss = " ".join(s.split()).strip()
        if not ss: 
            continue
        # toss if the whole thing is junk or extremely generic
        if ss.lower() in junk:
            continue
        # toss very long sentences
        if len(ss.split()) > 6:
            continue
        key = ss.lower()
        if key not in seen:
            seen.add(key)
            merged.append(ss)

    # 3) rank: give small bonus to LLM-produced strings
    ranked = _rank_terms(merged, llm_bonus=set(llm_out))
    # 4) cap to a tidy size
    top = ranked[:12]
    # ensure we keep at least 8 if available
    if len(top) < min(8, len(ranked)):
        top = ranked[:max(8, len(top))]
    return top


def _get_store():
    """Get the cached LanceMemory store."""
    return get_store()

def _gather_topic_evidence(store, task: str, *, retriever_model: str) -> List[Dict[str,Any]]:
    """
    Dynamic, topic-focused retrieval with more context (but still guarded against drift):
      - 6–10 dynamic terms
      - substring search window=3, limit per term=40
      - score by term hits + mild recency + slight preference for non-me (hearsay penalty)
      - pick top 4 chats by aggregate score
      - for each chosen chat, add a semantic top-up (k=24) restricted to that chat
      - de-dupe and cap total at 120 rows
      - carry both 'text' and 'clean_text' (composer uses clean_text; reply/rationale are cleaned later)
    """
    # --- step 1: term extraction (larger, still precise)
    terms = _gen_terms_llm(task, model=retriever_model)
    terms = terms[:10]

    window = 3
    limit_per_term = 40

    print("Terms: ", terms)

    prelim_rows: List[Dict[str,Any]] = []
    for term in terms:
        try:
            out = store.text_search(query=term, limit=limit_per_term, window=window)
        except Exception:
            continue
        for chat, msgs in out.items():
            for m in msgs:
                sender = (m.get("sender_name") or m.get("sender") or "").strip()
                is_me = (sender.lower() == "you")
                raw_text = (m.get("text") or "")
                clean_text = _clean_text_artifacts(raw_text)
                if not clean_text:
                    continue
                prelim_rows.append({
                    "chat": chat,
                    "sender": sender,
                    "sender_is_me": is_me,
                    "text": raw_text,
                    "clean_text": clean_text,
                    "timestamp": m.get("timestamp",""),
                })
    
    if not prelim_rows:
        return []

    # --- step 2: score by term anchors, recency, and non-me preference
    term_lc = [t.lower() for t in terms]
    def _t_hits(s: str) -> int:
        sl = s.lower()
        return sum(1 for t in term_lc if t in sl)

    def _recency_boost(ts_iso: str) -> float:
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(ts_iso.replace("Z","+00:00"))
            age_days = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()/86400.0
            return max(0.0, 1.0 - min(45.0, age_days)/45.0)
        except Exception:
            return 0.0

    for r in prelim_rows:
        anchors = _t_hits(r["clean_text"])
        r["_anchors"] = anchors
        r["_score"] = anchors + 0.18*_recency_boost(r["timestamp"]) + (0.12 if not r["sender_is_me"] else 0.0)

    # --- step 3: keep the most relevant chats (top 4 by summed score)
    by_chat: Dict[str, float] = {}
    for r in prelim_rows:
        by_chat[r["chat"]] = by_chat.get(r["chat"], 0.0) + r["_score"]
    top_chats = {c for c,_ in sorted(by_chat.items(), key=lambda kv: kv[1], reverse=True)[:4]}

    seeds = [r for r in prelim_rows if r["chat"] in top_chats]

    # --- step 4: semantic top-up per selected chat, constrained to that chat
    # this catches paraphrases that literal terms might miss, but *only* inside chosen chats.
    sem_rows: List[Dict[str,Any]] = []
    try:
        for chat in top_chats:
            for h in store.search(task, k=24, chat=chat):
                sender = (h.get("sender") or "").strip()
                is_me = (sender.lower() == "you")
                raw_text = (h.get("text") or "")
                clean_text = _clean_text_artifacts(raw_text)
                if not clean_text:
                    continue
                sem_rows.append({
                    "chat": chat,
                    "sender": sender,
                    "sender_is_me": is_me,
                    "text": raw_text,
                    "clean_text": clean_text,
                    "timestamp": h.get("timestamp",""),
                    "_anchors": _t_hits(clean_text),
                    "_score": 0.22,  # small uniform bump; literal anchors still dominate
                })
    except Exception:
        pass

    # --- step 5: merge, de-dupe, rank, cap
    merged: Dict[tuple, Dict[str,Any]] = {}
    for r in seeds + sem_rows:
        key = (r.get("chat"), r.get("timestamp"), r.get("text"))
        if key not in merged:
            merged[key] = r
        else:
            # keep the higher score if duplicate lands from both retrievals
            if r.get("_score",0.0) > merged[key].get("_score",0.0):
                merged[key] = r

    rows = list(merged.values())

    # heavy preference for true anchors (messages that actually contain a term)
    rows.sort(key=lambda r: (-(r.get("_anchors",0)), -(r.get("_score",0.0)), r.get("timestamp","")))
    return rows[:120]



SUGGEST_SYSTEM_PROMPT_V2 = """You must answer based ONLY on EVIDENCE_JSON.

Each evidence item has: chat, sender, sender_is_me, text, clean_text, timestamp.

GENERAL CONDUCT
- Use ONLY 'clean_text' from evidence when quoting or paraphrasing.
- NEVER include timestamps, “streamtyped”, UUIDs, or parser artifacts in your output.
- NEVER infer the speaker from the chat title. Use 'sender' and 'sender_is_me' only.
  • If sender_is_me=true, that line is from the prompter (not the other person).
- Keep outputs concise and human: 1–2 sentences for direct answers; short but complete pasteable replies.

QUALIFIER & TOPIC DISAMBIGUATION
- Treat qualified events as distinct (e.g., “Canadian Thanksgiving” vs “American Thanksgiving”).
- If the TASK specifies a qualifier, ignore evidence that refers clearly to a different qualifier unless it's used to EXPLAIN a contradiction (e.g., “can’t do Canadian but can do American”).
- Do not mix cities/venues unless evidence directly ties them to the same plan.

SPEAKER RELIABILITY & CONFLICTS
- Prefer direct statements from the named person over hearsay.
- Treat 'You' as hearsay unless 'You' are reporting a direct quote (“X said …”) and that quote is present.
- If evidence conflicts:
  • Prefer more direct/explicit statements over vague ones.
  • Prefer messages that explicitly mention the qualifier (e.g., “Canadian”) over ones that don’t.
  • Use ordering only for reasoning (do not output times): a later direct statement about the SAME qualified topic can supersede earlier speculation.

MODE SELECTION
- If the TASK is a QUESTION to the prompter: answer the prompter directly (not a paste message).
  • Also include a brief plain-language reason referencing at most two very short quotes (using clean_text only, no timestamps).
- If the TASK asks for a PASTEABLE REPLY to send: produce a clean, copy-ready message in the user's voice.
  • Where the exact answer is unknown, say you’ll confirm or propose a concrete next step (time, place, follow-up).

DECISION POLICY
- Output a 'verdict': "yes", "no", or "unknown".
  • "yes" only if evidence clearly supports the exact qualified scenario being asked.
  • "no" if evidence clearly contradicts it (e.g., “can’t come Canadian”).
  • "unknown" if evidence is insufficient, off-qualifier, or contradictory without resolution.

CITATION STYLE FOR RATIONALE (NOT IN REPLY)
- In the rationale, you may include up to two mini-quotes like: ["<chat> — <sender>": "<short clean quote>"].
- Do NOT include timestamps or artifacts.

RETURN STRICT JSON:
{
  "verdict": "yes" | "no" | "unknown",
  "reply": "concise answer to prompter OR pasteable message (no timestamps/artifacts)",
  "rationale": "1–2 sentence plain-language reason with up to two mini quotes (no timestamps)"
}
"""

def suggest_next_message_from_prompt(
    user_prompt: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 220,
) -> dict:
    from openai import OpenAI
    client = OpenAI()
    store = _get_store()  # cached LanceMemory; DB details remain internal

    # 1) low-volume, cleaned evidence
    evidence_items = _gather_topic_evidence(store, user_prompt, retriever_model=model)

    # 2) Build compact evidence payload
    evidence_json = [
        {
            "chat": r.get("chat",""),
            "sender": r.get("sender",""),
            "sender_is_me": bool(r.get("sender_is_me", False)),
            "text": r.get("text","")[:500],            # keep raw for completeness
            "clean_text": r.get("clean_text","")[:500],# composer will only use this
            "timestamp": r.get("timestamp",""),        # provided but MUST NOT be used in output
        }
        for r in evidence_items
    ]

    # 3) Compose with strict instructions
    composer_messages = [
        {"role": "system", "content": SUGGEST_SYSTEM_PROMPT_V2},
        {"role": "user", "content": json.dumps({"TASK": user_prompt, "EVIDENCE_JSON": evidence_json}, ensure_ascii=False)},
    ]

    resp = client.chat.completions.create(
        model=model,
        #temperature=temperature,
        # keep the same param style you used in the old version:
        max_completion_tokens=max_tokens,
        messages=composer_messages,
    )
    raw = (resp.choices[0].message.content or "{}").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"verdict":"unknown","reply":"","rationale":"Could not parse model output."}

    # 4) Text-friendly final reply
    reply = _text_friendly(parsed.get("reply",""))
    rationale = _text_friendly(parsed.get("rationale",""))
    prompter_answer = _is_question(user_prompt)

    # Append a short reason only for Q&A prompts
    if prompter_answer and rationale:
        # keep it short; avoid double punctuation
        sep = " — " if not reply.endswith((".", "!", "?")) else " "
        reply = f"{reply}{sep}reason: {rationale}"

    candidate_chats = list({e["chat"] for e in evidence_json if e.get("chat")})[:2]
    return {
        "reply": reply,
        "rationale": rationale,            # full rationale (already cleaned)
        "verdict": parsed.get("verdict","unknown"),
        "used": {
            "model": model,
            "max_tokens": max_tokens,
            "reply_mode": "prompter_answer" if prompter_answer else "pasteable_reply",
            "candidate_chats": candidate_chats,
            "evidence_count": len(evidence_json),
            "dynamic_terms": True,
        },
        "context": { "evidence_json": evidence_json }
    }

def retrieve_text(
    query: str,
    limit: int = 20,
    chat: Optional[str] = None,
    sender: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    window: int = 0
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve messages using text search functionality.
    Returns messages grouped by chat that contain the query text.
    """
    store = _get_store()
    return store.text_search(
        query=query,
        limit=limit,
        chat=chat,
        sender=sender,
        since=since,
        until=until,
        window=window
    )

if __name__ == "__main__":
    main()
