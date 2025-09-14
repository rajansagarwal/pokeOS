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
DEFAULT_CONTACTS  = os.path.join(os.path.dirname(__file__), "lib", ".contacts_cache.txt")
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

# --- lightweight module cache for the Lance store so we don't re-probe embeddings each call ---
_STORE: Optional[LanceMemory] = None
def _get_store() -> LanceMemory:
    global _STORE
    if _STORE is None:
        _STORE = LanceMemory(
            db_dir=DEFAULT_LANCEDB_DIR,
            table_name=DEFAULT_LANCEDB_TABLE,
            embedder=OpenAIEmbedder(),
        )
    return _STORE

# ===== Delete the old TOPIC_CANON_TERMS and _extract_topic_terms =====
# (remove the whole TOPIC_CANON_TERMS dict and any function that depended on it)

import json, re
from typing import List, Dict, Any, Set

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
