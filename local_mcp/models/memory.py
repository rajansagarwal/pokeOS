from __future__ import annotations
import os, re, datetime
from typing import Any, Dict, List, Optional
from ..lib.config import DEFAULT_LANCEDB_DIR, DEFAULT_LANCEDB_TABLE
from .embedder import OpenAIEmbedder

def _escape_sql(s: str) -> str:
    return s.replace("'", "''") if s else s

def _soft_sim_from_distance(d: Optional[float]) -> float:
    if d is None: return 0.0
    return 1.0 / (1.0 + float(d))

class LanceMemory:
    """
    On-disk vector DB (LanceDB) with OpenAI embeddings only.

    Schema:
      id: string (primary key)
      text: string
      chat: string
      sender: string
      timestamp: string (ISO)
      tags: list<string>
      vector: vector<float>
    """
    def __init__(
        self,
        db_dir: str = DEFAULT_LANCEDB_DIR,
        table_name: str = DEFAULT_LANCEDB_TABLE,
        embedder: Optional[OpenAIEmbedder] = None,
    ):
        import lancedb  # pip install lancedb
        import pathlib
        self.db_path = os.path.expanduser(db_dir)
        pathlib.Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        self.table_name = table_name
        self.embedder = embedder or OpenAIEmbedder()
        self.tbl = self._ensure_table()

    # ---------- TEXT SEARCH ----------
    def text_search(
        self,
        query: str,
        limit: int = 50,
        chat: Optional[str] = None,
        sender: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        case_insensitive: bool = True,
        window: int = 0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        import pandas as pd
        df = self.tbl.to_pandas().copy()

        for col in ["id", "text", "chat", "sender", "timestamp"]:
            if col not in df.columns:
                df[col] = None

        df["text"] = df["text"].fillna("").astype(str)
        df["chat"] = df["chat"].fillna("").astype(str)
        df["sender"] = df["sender"].fillna("").astype(str)
        df["__ts"] = pd.to_datetime(df.get("timestamp", None), errors="coerce", utc=True)

        if chat:
            df = df[df["chat"] == chat]
        if sender:
            df = df[df["sender"] == sender]
        if since:
            since_ts = pd.to_datetime(since, errors="coerce", utc=True)
            if not pd.isna(since_ts):
                df = df[df["__ts"] >= since_ts]
        if until:
            until_ts = pd.to_datetime(until, errors="coerce", utc=True)
            if not pd.isna(until_ts):
                df = df[df["__ts"] <= until_ts]

        mask = df["text"].str.contains(query, case=not case_insensitive, regex=False, na=False)

        if window <= 0:
            hits_df = df[mask].copy().sort_values("__ts", ascending=False).head(limit)
            out: Dict[str, List[Dict[str, Any]]] = {}
            for _, r in hits_df.iterrows():
                out.setdefault(r["chat"], []).append({
                    "sender_name": r["sender"] or "Other",
                    "text": r["text"] or "",
                    "timestamp": str(r.get("timestamp") or ""),
                })
            for k in out:
                out[k] = sorted(out[k], key=lambda x: x["timestamp"])
            return out

        df = df.sort_values(["chat", "__ts"], ascending=[True, True])
        df["__pos"] = df.groupby("chat").cumcount()
        hits_df = df[mask].copy().sort_values("__ts", ascending=False).head(limit)

        wanted_idxs = set()
        for _, r in hits_df.iterrows():
            c = r["chat"]; p = int(r["__pos"])
            seg = df[df["chat"] == c]
            lo, hi = p - window, p + window
            wanted_idxs.update(seg[(seg["__pos"] >= lo) & (seg["__pos"] <= hi)].index.tolist())

        ctx_df = df.loc[sorted(wanted_idxs)].copy()
        out: Dict[str, List[Dict[str, Any]]] = {}
        for chat_name, grp in ctx_df.groupby("chat", sort=False):
            msgs = []
            for _, r in grp.iterrows():
                msgs.append({
                    "sender_name": r["sender"] or "Other",
                    "text": r["text"] or "",
                    "timestamp": str(r.get("timestamp") or ""),
                })
            out[chat_name] = msgs
        return out

    def _ensure_table(self):
        import pyarrow as pa
        dim = len(self.embedder.embed_texts(["_probe_"])[0])
        existing = set(self.db.table_names())
        if self.table_name in existing:
            return self.db.open_table(self.table_name)

        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("chat", pa.string()),
            pa.field("sender", pa.string()),
            pa.field("timestamp", pa.string()),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ])

        seed = [{
            "id": "seed",
            "text": "seed",
            "chat": "system",
            "sender": "system",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "tags": ["seed"],
            "vector": [0.0] * dim,
        }]

        return self.db.create_table(self.table_name, data=seed, schema=schema, mode="overwrite")

    def _clean_text(self, s: str) -> str:
        import re as _re
        s = (s or "").strip()
        s = _re.sub(r"\s+", " ", s)
        return s

    def existing_ids(self) -> set[str]:
        try:
            df = self.tbl.to_pandas()
            if "id" in df.columns:
                return set(df["id"].astype(str).tolist())
        except Exception:
            pass
        try:
            at = self.tbl.to_arrow()
            try:
                arr = at["id"]
            except Exception:
                idx = at.schema.get_field_index("id")
                if idx == -1: return set()
                arr = at.column(idx)
            return set(str(x) for x in arr.to_pylist())
        except Exception:
            pass
        try:
            rows = self.tbl.head(1_000_000)
            if isinstance(rows, list):
                return set(str(r.get("id")) for r in rows if isinstance(r, dict) and "id" in r)
        except Exception:
            pass
        return set()

    def upsert(self, rows: List[Dict[str, Any]], skip_existing: bool = True) -> int:
        if not rows: return 0
        existing = self.existing_ids() if skip_existing else set()

        cleaned, texts = [], []
        for r in rows:
            rid = str(r["id"])
            if skip_existing and rid in existing:
                continue
            txt = self._clean_text(r.get("text",""))
            if not txt: continue
            cleaned.append({
                "id": rid,
                "text": txt,
                "chat": r.get("chat",""),
                "sender": r.get("sender",""),
                "timestamp": r.get("timestamp") or datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "tags": list(r.get("tags") or []),
            })
            texts.append(txt)

        if not cleaned: return 0
        vecs = self.embedder.embed_texts(texts)
        for rr, v in zip(cleaned, vecs):
            rr["vector"] = v
        self.tbl.add(cleaned, mode="overwrite")
        return len(cleaned)

    def search(
        self,
        query: str,
        k: int = 12,
        chat: Optional[str] = None,
        sender: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        tags_any: Optional[List[str]] = None,
    ) -> List[Dict[str,Any]]:
        qvec = self.embedder.embed_texts([query])[0]
        search = self.tbl.search(qvec).limit(k)
        filters = []
        if chat:   filters.append(f"chat = '{_escape_sql(chat)}'")
        if sender: filters.append(f"sender = '{_escape_sql(sender)}'")
        if since:  filters.append(f"timestamp >= '{_escape_sql(since)}'")
        if until:  filters.append(f"timestamp <= '{_escape_sql(until)}'")
        if tags_any:
            ors = " OR ".join([f"list_has(tags, '{_escape_sql(t)}')" for t in tags_any])
            filters.append(f"({ors})")
        if filters:
            search = search.where(" AND ".join(filters))
        hits = search.to_list()
        out = []
        for h in hits:
            out.append({
                "id": h.get("id"),
                "text": h.get("text"),
                "chat": h.get("chat"),
                "sender": h.get("sender"),
                "timestamp": h.get("timestamp"),
                "tags": h.get("tags", []),
                "score": _soft_sim_from_distance(h.get("_distance"))
            })
        return out

    def relevant_context(self, prompt: str, k_per_thread: int = 6, max_threads: int = 3) -> List[Dict[str,Any]]:
        broad = self.search(prompt, k=50)
        by_chat: Dict[str, List[Dict[str,Any]]] = {}
        for r in broad:
            by_chat.setdefault(r["chat"], []).append(r)

        def age_hours(ts: str) -> float:
            try:
                dt = datetime.datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                return (datetime.datetime.now(datetime.timezone.utc) - dt.astimezone(datetime.timezone.utc)).total_seconds()/3600.0
            except Exception:
                return 1e9

        ranked: List[tuple[str,float]] = []
        for chat, items in by_chat.items():
            items.sort(key=lambda x: x["score"], reverse=True)
            top = items[:k_per_thread*2]
            rec = sum(1.0/(1.0+age_hours(x["timestamp"])/24.0) for x in top)
            score = sum(x["score"] for x in top) + 0.5*rec
            ranked.append((chat, score))
        ranked.sort(key=lambda t: t[1], reverse=True)
        chosen = ranked[:max_threads]

        out=[]
        for chat,_ in chosen:
            items = sorted(by_chat[chat], key=lambda x: x["timestamp"])
            items = items[-k_per_thread:]
            out.append({
                "chat": chat,
                "messages": [{"sender":i["sender"],"text":i["text"],"timestamp":i["timestamp"]} for i in items]
            })
        return out

# --- lightweight cached store so we don't re-probe embeddings each call ---
_STORE: Optional[LanceMemory] = None
def get_store(db_dir: Optional[str] = None, table_name: Optional[str] = None) -> LanceMemory:
    global _STORE
    if _STORE is None or (db_dir or table_name):
        _STORE = LanceMemory(
            db_dir=db_dir or DEFAULT_LANCEDB_DIR,
            table_name=table_name or DEFAULT_LANCEDB_TABLE,
            embedder=OpenAIEmbedder(),
        )
    return _STORE
