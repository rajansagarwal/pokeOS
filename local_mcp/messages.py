#!/usr/bin/env python3
from __future__ import annotations
import os, re, json, sqlite3, shutil, datetime, argparse, pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

# Load environment variables from .env (project root or parent dirs)
load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("Set OPENAI_API_KEY in .env")

# ================== Config ==================
DEFAULT_DB_PATH   = os.path.expanduser("~/Library/Messages/chat.db")
SNAPSHOT_DIR      = os.path.expanduser("~/Library/Messages/_snapshots")
DEFAULT_CONTACTS  = os.path.expanduser("~/.contacts_cache.txt")
RECENT_CHATS      = 20
MESSAGES_PER_CHAT = 30

# LanceDB defaults
DEFAULT_LANCEDB_DIR   = os.path.expanduser("~/.chat_memdb")
DEFAULT_LANCEDB_TABLE = "messages"

LOCAL_TZ    = datetime.datetime.now().astimezone().tzinfo
APPLE_EPOCH = datetime.datetime(2001, 1, 1, tzinfo=datetime.timezone.utc)

# ================== Time ====================
def apple_time_to_dt(raw: Optional[int|float]) -> Optional[datetime.datetime]:
    if raw is None: return None
    try: val = int(raw)
    except: return None
    seconds = val / 1_000_000_000 if abs(val) > 10**12 else val
    return (APPLE_EPOCH + datetime.timedelta(seconds=seconds)).astimezone(LOCAL_TZ)

def dt_to_iso(dt: Optional[datetime.datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None

# ============ Phone normalization ============
_PHONE_URI_PREFIX_RE = re.compile(r"^(?:tel:|sms:)", re.I)
_EXT_TAIL_RE         = re.compile(r"(?:ext\.?|x)\s*\d+\s*$", re.I)

def clean_phone_to_digits(s: Optional[str]) -> str:
    """Remove tel:/sms:, spaces, brackets, dashes, plus, dots, and simple extensions → digits only."""
    if not s: return ""
    s = s.strip()
    s = _PHONE_URI_PREFIX_RE.sub("", s)
    s = _EXT_TAIL_RE.sub("", s)
    return re.sub(r"\D", "", s)

def canonical_phone_key(s: Optional[str]) -> str:
    """Use last 10 digits (NANP) to compare; else whatever digits remain."""
    d = clean_phone_to_digits(s)
    return d[-10:] if len(d) >= 10 else d

def phone_match_keys(s: Optional[str]) -> List[str]:
    """Variants we’ll try when resolving a handle against the cache."""
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

# ============ Contacts cache loader ==========
def load_contacts_cache(path: str) -> tuple[Dict[str,str], Dict[str,str]]:
    """
    Parse a file with lines like:
      Name||phone||+1 (425) 777-0183
      Name||email||foo@bar.com
    Build two indexes:
      phone_index: last10/full/last7 digits -> name
      email_index: lowercased email -> name
    """
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
            name = name.strip()
            value = value.strip()
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

# ============ DB snapshot & open ============
def snapshot_db(src: str = DEFAULT_DB_PATH, out_dir: str = SNAPSHOT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Messages DB not found: {src}")
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = os.path.join(out_dir, f"chat-{ts}.db")
    try:
        src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        dst_conn = sqlite3.connect(dst)
        src_conn.backup(dst_conn)
        dst_conn.close(); src_conn.close()
    except Exception:
        shutil.copy2(src, dst)
    # best-effort wal/shm
    base = os.path.dirname(src)
    for sfx in ("-wal", "-shm"):
        p = os.path.join(base, f"chat.db{sfx}")
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out_dir, f"chat-{ts}.db{sfx}"))
    return dst

def open_db_ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)

# ============ Basic models & queries ============
@dataclass
class ChatRow:
    chat_id: int
    guid: str
    display_name: Optional[str]
    last_date: Optional[int]

def list_recent_chats(conn: sqlite3.Connection, limit: int) -> List[ChatRow]:
    q = """
    SELECT c.ROWID, c.guid, c.display_name, MAX(m.date) as last_date
    FROM chat c
    JOIN chat_message_join cmj ON cmj.chat_id = c.ROWID
    JOIN message m ON m.ROWID = cmj.message_id
    GROUP BY c.ROWID
    ORDER BY last_date DESC
    LIMIT ?
    """
    cur = conn.cursor()
    cur.execute(q, (limit,))
    return [ChatRow(*row) for row in cur.fetchall()]

def chat_participants(conn: sqlite3.Connection, chat_id: int) -> List[str]:
    q = """
    SELECT h.id
    FROM chat_handle_join chj
    JOIN handle h ON h.ROWID = chj.handle_id
    WHERE chj.chat_id = ?
    ORDER BY h.id
    """
    cur = conn.cursor()
    cur.execute(q, (chat_id,))
    return [r[0] for r in cur.fetchall()]

def recent_messages_for_chat(conn: sqlite3.Connection, chat_id: int, limit: int) -> List[Dict[str,Any]]:
    q = """
    SELECT
      m.ROWID,
      m.is_from_me,
      m.service,
      m.text,
      m.attributedBody,
      m.date,
      m.handle_id
    FROM chat_message_join cmj
    JOIN message m ON m.ROWID = cmj.message_id
    WHERE cmj.chat_id = ?
    ORDER BY m.date DESC
    LIMIT ?
    """
    cur = conn.cursor()
    cur.execute(q, (chat_id, limit))
    rows = cur.fetchall()
    q_sender = "SELECT id FROM handle WHERE ROWID = ?"

    out: List[Dict[str,Any]] = []
    for (rowid, is_from_me, service, text, attrib, date_raw, handle_rowid) in rows:
        sender_handle = "me" if is_from_me == 1 else None
        if sender_handle is None and handle_rowid is not None:
            cur.execute(q_sender, (handle_rowid,))
            r = cur.fetchone()
            sender_handle = r[0] if r else None

        msg_text = text or ""
        if not msg_text and attrib:
            if isinstance(attrib, memoryview): attrib = attrib.tobytes()
            elif not isinstance(attrib, (bytes, bytearray)):
                try: attrib = bytes(attrib)
                except Exception: attrib = None
            if attrib:
                # quick fallback decode for attributedBody
                parts = re.findall(rb"[ -~]{4,}", attrib)
                if parts:
                    t = " ".join(p.decode("utf-8", "ignore") for p in parts)
                    t = re.sub(r"\s+", " ", t)
                    t = re.sub(r"NS[A-Za-z]+|NSDictionary|NSNumber|NSValue|__kIM\w+", "", t)
                    t = re.sub(r"\s{2,}", " ", t).strip()
                    if t: msg_text = t

        out.append({
            "id": int(rowid),
            "is_from_me": bool(is_from_me == 1),
            "sender_handle": sender_handle,
            "text": msg_text,
            "timestamp": dt_to_iso(apple_time_to_dt(date_raw)),
            "service": service
        })
    return list(reversed(out))  # oldest→newest for LLM friendliness

# ============ Display name for thread ============
def display_name_for_chat(chat: ChatRow, participants: List[str],
                          phone_idx: Dict[str,str], email_idx: Dict[str,str]) -> str:
    if chat.display_name:
        return chat.display_name
    if len(participants) == 1:
        h = participants[0]
        return resolve_name(h, phone_idx, email_idx) or h
    names = [(resolve_name(h, phone_idx, email_idx) or h) for h in participants]
    return ", ".join(names) if names else chat.guid

# ============ High-level JSON (MCP-ready) ============
def read_recent_conversations_json(contacts_path: str = DEFAULT_CONTACTS,
                                   chats_limit: int = RECENT_CHATS,
                                   per_chat_limit: int = MESSAGES_PER_CHAT) -> Dict[str, List[Dict[str,Any]]]:
    """
    Returns: { "<Person or Group Name>": [ {sender_name, text, timestamp}, ... ], ... }
    """
    phone_idx, email_idx = load_contacts_cache(contacts_path)
    snap = snapshot_db(DEFAULT_DB_PATH, SNAPSHOT_DIR)
    conn = open_db_ro(snap)
    try:
        chats = list_recent_chats(conn, chats_limit)
        result: Dict[str, List[Dict[str,Any]]] = {}
        for ch in chats:
            participants = chat_participants(conn, ch.chat_id)
            disp = display_name_for_chat(ch, participants, phone_idx, email_idx)
            msgs = recent_messages_for_chat(conn, ch.chat_id, per_chat_limit)
            simplified = []
            for m in msgs:
                if m.get("is_from_me"):
                    sender_name = "You"
                else:
                    sender_name = resolve_name(m.get("sender_handle"), phone_idx, email_idx) \
                                or (m.get("sender_handle") or "Other")
                simplified.append({
                    "sender_name": sender_name,
                    "text": m.get("text") or "",
                    "timestamp": m.get("timestamp"),
                })
            result[disp] = simplified
        return result
    finally:
        conn.close()

# --------------------------------------------------------------------------
#                          OpenAI-only embeddings + LanceDB
# --------------------------------------------------------------------------
class OpenAIEmbedder:
    """
    OpenAI embeddings (text-embedding-3-small by default).
    Requires: pip install openai  AND  OPENAI_API_KEY env var.
    """
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use OpenAI embeddings.")
        try:
            from openai import OpenAI  # import only when requested
        except ModuleNotFoundError as e:
            raise RuntimeError("The 'openai' package is not installed. Run: pip install openai") from e
        self._OpenAI = OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

# ------------ LanceDB store ------------
def _escape_sql(s: str) -> str:
    return s.replace("'", "''") if s else s

def _soft_sim_from_distance(d: Optional[float]) -> float:
    if d is None: return 0.0
    return 1.0 / (1.0 + float(d))  # higher is better

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
    def __init__(self,
                 db_dir: str = DEFAULT_LANCEDB_DIR,
                 table_name: str = DEFAULT_LANCEDB_TABLE,
                 embedder: Optional[OpenAIEmbedder] = None):
        import lancedb  # pip install lancedb
        self.db_path = os.path.expanduser(db_dir)
        pathlib.Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        self.table_name = table_name
        self.embedder = embedder or OpenAIEmbedder()
        self.tbl = self._ensure_table()

    # ---------- TEXT SEARCH (reads from LanceDB table) ----------
    def text_search(
        self,
        query: str,
        limit: int = 50,
        chat: Optional[str] = None,
        sender: Optional[str] = None,
        since: Optional[str] = None,  # ISO string (date or datetime)
        until: Optional[str] = None,  # ISO string (date or datetime)
        case_insensitive: bool = True,
        window: int = 0,              # include N messages before/after each hit within the same chat
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Literal substring search over the Lance table (iMessage-style), with simple filters.
        Returns { chat_name: [ {sender_name, text, timestamp}, ... ], ... }.
        """
        import pandas as pd

        # Pull the table into pandas (works with older lancedb versions; no columns= kw)
        df = self.tbl.to_pandas().copy()

        # Basic schema guards
        for col in ["id", "text", "chat", "sender", "timestamp"]:
            if col not in df.columns:
                df[col] = None

        # Normalize types
        df["text"] = df["text"].fillna("").astype(str)
        df["chat"] = df["chat"].fillna("").astype(str)
        df["sender"] = df["sender"].fillna("").astype(str)

        # Parse timestamp column to pandas datetime (coerce invalid)
        df["__ts"] = pd.to_datetime(df.get("timestamp", None), errors="coerce", utc=True)

        # Apply filters
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

        # Literal substring match (regex=False). case parameter controls case sensitivity.
        mask = df["text"].str.contains(query, case=not case_insensitive, regex=False, na=False)
        hits_df = df[mask].copy()

        # If no context window, return top-N newest hits grouped by chat
        if window <= 0:
            hits_df = hits_df.sort_values("__ts", ascending=False).head(limit)
            out: Dict[str, List[Dict[str, Any]]] = {}
            for _, r in hits_df.iterrows():
                out.setdefault(r["chat"], []).append({
                    "sender_name": r["sender"] or "Other",
                    "text": r["text"] or "",
                    "timestamp": str(r.get("timestamp") or ""),
                })
            # Keep messages oldest→newest for each chat (nicer to read)
            for k in out:
                out[k] = sorted(out[k], key=lambda x: x["timestamp"])
            return out

        # With a window, gather neighboring rows per chat
        # Sort per chat by time to make window slicing intuitive
        df = df.sort_values(["chat", "__ts"], ascending=[True, True])
        # Create per-chat integer positions
        df["__pos"] = df.groupby("chat").cumcount()

        # Identify positions of the hits
        hits_df = df[mask].copy()
        # Limit the number of hit-centers first (newest first) to bound the total output size
        hits_df = hits_df.sort_values("__ts", ascending=False).head(limit)

        # For each hit, collect +/- window rows in the same chat by position
        wanted_idxs = set()
        for _, r in hits_df.iterrows():
            c = r["chat"]
            p = int(r["__pos"])
            # slice this chat’s segment
            seg = df[df["chat"] == c]
            # find indices within window range by position
            lo, hi = p - window, p + window
            wanted_idxs.update(seg[(seg["__pos"] >= lo) & (seg["__pos"] <= hi)].index.tolist())

        ctx_df = df.loc[sorted(wanted_idxs)].copy()

        # Build grouped output (keep chronological order inside each chat)
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

        # probe the embedding dimension once
        dim = len(self.embedder.embed_texts(["_probe_"])[0])

        existing = set(self.db.table_names())
        if self.table_name in existing:
            return self.db.open_table(self.table_name)

        # Explicit schema: strings, list<string> for tags, and vector as fixed_size_list<float32>[dim]
        schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("chat", pa.string()),
                pa.field("sender", pa.string()),
                pa.field("timestamp", pa.string()),
                pa.field("tags", pa.list_(pa.string())),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ])

        # Seed row with NON-empty tags so Arrow sees list<string>, not list<null>.
        seed = [{
            "id": "seed",
            "text": "seed",
            "chat": "system",
            "sender": "system",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "tags": ["seed"],  # <- important: not []
            "vector": [0.0] * dim,
        }]

        # Some LanceDB versions want data + schema; others accept just data.
        # Providing both is safest; 'mode="overwrite"' creates if missing.
        return self.db.create_table(
            self.table_name,
            data=seed,
            schema=schema,
            mode="overwrite",
        )

    def _clean_text(self, s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        return s
    
    def existing_ids(self) -> set[str]:
        # Try Pandas (most convenient)
        try:
            import pandas as pd  # noqa: F401
            df = self.tbl.to_pandas()              # no 'columns' kw in older versions
            if "id" in df.columns:
                return set(df["id"].astype(str).tolist())
        except Exception:
            pass

        # Try Arrow directly
        try:
            at = self.tbl.to_arrow()               # pyarrow.Table
            try:
                arr = at["id"]                     # column by name (newer pyarrow)
            except Exception:
                idx = at.schema.get_field_index("id")
                if idx == -1:
                    return set()
                arr = at.column(idx)               # column by index (older pyarrow)
            return set(str(x) for x in arr.to_pylist())
        except Exception:
            pass

        # Last resort: scan head (limited)
        try:
            rows = self.tbl.head(1_000_000)        # some versions return list[dict]
            if isinstance(rows, list):
                return set(str(r.get("id")) for r in rows if isinstance(r, dict) and "id" in r)
        except Exception:
            pass

        return set()

    def upsert(self, rows: List[Dict[str, Any]], skip_existing: bool = True) -> int:
        if not rows:
            return 0

        # Optionally filter out rows whose ids are already in the table
        existing = self.existing_ids() if skip_existing else set()

        cleaned, texts, ids = [], [], []
        for r in rows:
            rid = str(r["id"])
            if skip_existing and rid in existing:
                continue  # already stored

            txt = self._clean_text(r.get("text",""))
            if not txt:
                continue

            cleaned.append({
                "id": rid,
                "text": txt,
                "chat": r.get("chat",""),
                "sender": r.get("sender",""),
                "timestamp": r.get("timestamp") or datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "tags": list(r.get("tags") or []),
            })
            texts.append(txt)
            ids.append(rid)

        if not cleaned:
            return 0

        # (If you're on older LanceDB without primary keys, do your delete-then-add here)
        # self._delete_ids(ids)

        vecs = self.embedder.embed_texts(texts)
        for rr, v in zip(cleaned, vecs):
            rr["vector"] = v

        # For primary-key tables (>=0.7), overwrite is fine; otherwise plain add()
        self.tbl.add(cleaned, mode="overwrite")
        return len(cleaned)

    def search(self,
               query: str,
               k: int = 12,
               chat: Optional[str] = None,
               sender: Optional[str] = None,
               since: Optional[str] = None,
               until: Optional[str] = None,
               tags_any: Optional[List[str]] = None) -> List[Dict[str,Any]]:
        qvec = self.embedder.embed_texts([query])[0]
        search = self.tbl.search(qvec).limit(k)
        filters = []
        if chat:  filters.append(f"chat = '{_escape_sql(chat)}'")
        if sender:filters.append(f"sender = '{_escape_sql(sender)}'")
        if since: filters.append(f"timestamp >= '{_escape_sql(since)}'")
        if until: filters.append(f"timestamp <= '{_escape_sql(until)}'")
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
        # broad pull
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

        ranked: List[Tuple[str,float]] = []
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

# ------------ Convert DB rows → Lance rows ------------
def read_recent_conversations_for_indexing(contacts_path: str = DEFAULT_CONTACTS,
                                           chats_limit: int = RECENT_CHATS,
                                           per_chat_limit: int = MESSAGES_PER_CHAT) -> List[Dict[str,Any]]:
    """
    Returns list of rows ready to index in LanceDB:
    [{id,text,chat,sender,timestamp,tags}]
    """
    phone_idx, email_idx = load_contacts_cache(contacts_path)
    snap = snapshot_db(DEFAULT_DB_PATH, SNAPSHOT_DIR)
    conn = open_db_ro(snap)
    rows: List[Dict[str,Any]] = []
    try:
        chats = list_recent_chats(conn, chats_limit)
        for ch in chats:
            participants = chat_participants(conn, ch.chat_id)
            disp = display_name_for_chat(ch, participants, phone_idx, email_idx)
            msgs = recent_messages_for_chat(conn, ch.chat_id, per_chat_limit)
            for m in msgs:
                sender_name = "You" if m.get("is_from_me") else (resolve_name(m.get("sender_handle"), phone_idx, email_idx) or (m.get("sender_handle") or "Other"))
                rows.append({
                    "id": str(m["id"]),
                    "text": m.get("text") or "",
                    "chat": disp,
                    "sender": sender_name,
                    "timestamp": m.get("timestamp"),
                    "tags": [m.get("service") or "Messages"]
                })
    finally:
        conn.close()
    return [r for r in rows if (r["text"] and r["timestamp"])]

# ============ CLI ============
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Read recent Messages; index & query with LanceDB (OpenAI embeddings only)."
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
    ap.add_argument("--window", type=int, default=0,
                    help="Include N messages of context before/after each text hit (default 0 = hits only)")

    args = ap.parse_args()

    # Default: dump recent conversations as JSON (no LanceDB needed)
    if not any([args.index, args.query, args.text, args.context]):
        data = read_recent_conversations_json(args.contacts, args.chats, args.per)
        print(json.dumps(data, ensure_ascii=False, indent=2 if args.pretty else None))
        raise SystemExit(0)

    # Open LanceDB (and embedder for vector ops)
    embedder = OpenAIEmbedder()
    store = LanceMemory(db_dir=args.dbdir, table_name=args.table, embedder=embedder)

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
        # Use the pure-Python text search over the Lance table (exact substring, case-insensitive)
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
