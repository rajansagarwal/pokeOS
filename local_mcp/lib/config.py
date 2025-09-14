import os, datetime

# Paths & limits
DEFAULT_DB_PATH   = os.path.expanduser("~/Library/Messages/chat.db")
SNAPSHOT_DIR      = os.path.expanduser("~/Library/Messages/_snapshots")
DEFAULT_CONTACTS  = os.path.expanduser("~/.contacts_cache.txt")
RECENT_CHATS      = 20
MESSAGES_PER_CHAT = 30

# LanceDB defaults
DEFAULT_LANCEDB_DIR   = os.path.expanduser("~/.chat_memdb")
DEFAULT_LANCEDB_TABLE = "messages"
