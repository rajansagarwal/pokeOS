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

if __name__ == "__main__":
    main()
