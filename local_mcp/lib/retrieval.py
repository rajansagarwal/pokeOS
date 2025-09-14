from __future__ import annotations
from typing import Any, Dict, List
from lib.text_utils import clean_text_artifacts
from lib.terms import gen_terms_llm
from lib.log import debug

def gather_topic_evidence(store, task: str, *, retriever_model: str) -> List[Dict[str,Any]]:
    """
    Dynamic, topic-focused retrieval with guarded expansion:
      - 6–10 dynamic terms
      - substring search window=3, limit per term=40
      - score by term hits + mild recency + slight preference for non-me (hearsay penalty)
      - pick top 4 chats by aggregate score
      - per-chat semantic top-up (k=24) restricted to chosen chats
      - de-dupe and cap total at 120 rows
      - include both 'text' and 'clean_text'
    """
    terms = gen_terms_llm(task, model=retriever_model)[:10]
    window = 3
    limit_per_term = 40
    debug(f"gather terms={terms}")

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
                clean_text = clean_text_artifacts(raw_text)
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

    by_chat: Dict[str, float] = {}
    for r in prelim_rows:
        by_chat[r["chat"]] = by_chat.get(r["chat"], 0.0) + r["_score"]
    top_chats = {c for c,_ in sorted(by_chat.items(), key=lambda kv: kv[1], reverse=True)[:4]}
    seeds = [r for r in prelim_rows if r["chat"] in top_chats]

    sem_rows: List[Dict[str,Any]] = []
    try:
        for chat in top_chats:
            for h in store.search(task, k=24, chat=chat):
                sender = (h.get("sender") or "").strip()
                is_me = (sender.lower() == "you")
                raw_text = (h.get("text") or "")
                clean_text = clean_text_artifacts(raw_text)
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
                    "_score": 0.22,
                })
    except Exception:
        pass

    merged: Dict[tuple, Dict[str,Any]] = {}
    for r in seeds + sem_rows:
        key = (r.get("chat"), r.get("timestamp"), r.get("text"))
        if key not in merged:
            merged[key] = r
        else:
            if r.get("_score",0.0) > merged[key].get("_score",0.0):
                merged[key] = r

    rows = list(merged.values())
    rows.sort(key=lambda r: (-(r.get("_anchors",0)), -(r.get("_score",0.0)), r.get("timestamp","")))
    return rows[:120]
