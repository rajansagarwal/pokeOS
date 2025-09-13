from __future__ import annotations
import os, re, json, sqlite3, shutil, datetime, argparse, pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# ================== Config ==================
DEFAULT_DB_PATH   = os.path.expanduser("~/Library/Messages/chat.db")
SNAPSHOT_DIR      = os.path.expanduser("~/Library/Messages/_snapshots")
DEFAULT_CONTACTS  = os.path.expanduser("~/.contacts_cache.txt")
RECENT_CHATS      = 20
MESSAGES_PER_CHAT = 30

LOCAL_TZ   = datetime.datetime.now().astimezone().tzinfo
APPLE_EPOCH= datetime.datetime(2001, 1, 1, tzinfo=datetime.timezone.utc)

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
    if len(d) >= 10: keys.append(d[-10:])  # preferred
    keys.append(d)                          # full digits as-is
    if len(d) >= 7:  keys.append(d[-7:])   # ultra fallback
    # de-dup preserve order
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
    if chat.display_name:  # Apple-provided group name
        return chat.display_name
    if len(participants) == 1:
        h = participants[0]
        return resolve_name(h, phone_idx, email_idx) or h
    names = [(resolve_name(h, phone_idx, email_idx) or h) for h in participants]
    return ", ".join(names) if names else chat.guid

# ============ High-level (MCP-ready) ============
def read_recent_conversations_json(contacts_path: str = DEFAULT_CONTACTS,
                                   chats_limit: int = RECENT_CHATS,
                                   per_chat_limit: int = MESSAGES_PER_CHAT) -> Dict[str, List[Dict[str,Any]]]:
    """
    Returns: { "<Person or Group Name>": [ {message...}, ... ], ... }
    """
    # Load contacts cache (no runtime Contacts access)
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

# ============ CLI ============
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Read recent Messages using a pre-dumped Contacts cache")
    ap.add_argument("--contacts", default=DEFAULT_CONTACTS, help="Path to contacts cache file")
    ap.add_argument("--chats", type=int, default=RECENT_CHATS, help="How many recent chats")
    ap.add_argument("--per", type=int, default=MESSAGES_PER_CHAT, help="How many messages per chat")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = ap.parse_args()

    data = read_recent_conversations_json(args.contacts, args.chats, args.per)
    print(json.dumps(data, ensure_ascii=False, indent=2 if args.pretty else None))
