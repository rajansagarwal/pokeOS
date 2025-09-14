from __future__ import annotations
import os, re, json, sqlite3, shutil, datetime, pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from ..lib.config import (
    DEFAULT_DB_PATH,
    SNAPSHOT_DIR,
    DEFAULT_CONTACTS,
    RECENT_CHATS,
    MESSAGES_PER_CHAT,
)
from ..lib.time_utils import apple_time_to_dt, dt_to_iso
from ..lib.contacts import load_contacts_cache, resolve_name

# -------- DB snapshot & open --------
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
    base = os.path.dirname(src)
    for sfx in ("-wal", "-shm"):
        p = os.path.join(base, f"chat.db{sfx}")
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out_dir, f"chat-{ts}.db{sfx}"))
    return dst

def open_db_ro(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)

# -------- Basic models & queries --------
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
    return list(reversed(out))  # oldest→newest

def display_name_for_chat(chat: ChatRow, participants: List[str],
                          phone_idx: Dict[str,str], email_idx: Dict[str,str]) -> str:
    if chat.display_name:
        return chat.display_name
    if len(participants) == 1:
        h = participants[0]
        return resolve_name(h, phone_idx, email_idx) or h
    names = [(resolve_name(h, phone_idx, email_idx) or h) for h in participants]
    return ", ".join(names) if names else chat.guid

# -------- High-level JSON (MCP-ready) --------
def read_recent_conversations_json(contacts_path: str = DEFAULT_CONTACTS,
                                   chats_limit: int = RECENT_CHATS,
                                   per_chat_limit: int = MESSAGES_PER_CHAT) -> Dict[str, List[Dict[str,Any]]]:
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

# -------- Convert DB rows → Lance rows --------
def read_recent_conversations_for_indexing(contacts_path: str = DEFAULT_CONTACTS,
                                           chats_limit: int = RECENT_CHATS,
                                           per_chat_limit: int = MESSAGES_PER_CHAT) -> List[Dict[str,Any]]:
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
