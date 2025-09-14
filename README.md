# poke-desktop

## Messages


Snapshot your local iMessage DB, index recent chats into LanceDB (OpenAI embeddings), and **answer questions or draft paste-ready replies**. 

The `local_mcp` directory contains a Python script that reads recent messages from the Messages database and converts them into a JSON format that can be used by the MCP. The script uses a contacts cache to resolve phone numbers and email addresses to names.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Permissions

You will need to make sure that your IDE/Terminal has Full Disk Access permissions to access the iMessage database. You can do this by going to System Settings > Security & Privacy > Privacy > Full Disk Access and adding your IDE/Terminal to the list.

Additionally, if you're using the Contacts Cache, you will need to make sure that your IDE/Terminal has access to your Contacts.

### Contacts Cache

To map phone/emails → names, create ~/.contacts_cache.txt using the `contacts_cache_dump.py` script located in the local_mcp/scripts directory. You must first run the following to store a local copy of your contacts on your filesystem.

```bash
python3 local_mcp/scripts/contacts_cache_dump.py
```

If missing, results still work, senders just appear as raw handles.

### First Run

Index recent iMessage history to LanceDB (with optional arguments):

```bash
python messages.py \
  --index \
  --chats 40 \
  --per 60 \
  --dbdir ~/.chat_memdb \
  --table messages
```

This snapshots `~/Library/Messages/chat.db` into `~/Library/Messages/_snapshots/` and upserts into `~/.chat_memdb`.

### CLI Usage

> Note: All commands accept --pretty for pretty-printed JSON.

a) Dump recent conversations (no DB needed)

```bash
python messages.py --pretty > recent.json
```

b) Plain substring search (exact, case-insensitive)

```bash
# Find "Thanksgiving" with small context window
python messages.py --text "Thanksgiving" --window 2 --pretty

# Filter by chat and time range
python messages.py \
  --text "Turkey" \
  --inchat "Random Groupchat" \
  --since 2025-09-01 \
  --until 2025-09-30 \
  --window 3 \
  --pretty
```

c) Semantic search (embedding similarity)

Query argument uses the prompt to fetch relevant messages of texts, e.g. a single message.
```bash
python messages.py --query "Thanksgiving plans with Jeff" --k 8 --pretty
```

d) Pull bounded context per thread

Context argument uses the prompt to fetch relevant threads of texts, e.g. chains of messages in a thread.

```bash
python messages.py --context "Am I free for Thanksgiving?" --ctx_k 6 --ctx_threads 3 --pretty
```

### Python API

Use the high-level function `suggest_next_message_from_prompt` to answer questions or generate a pasteable reply.

```python
from local_mcp.messages import suggest_next_message_from_prompt

result = suggest_next_message_from_prompt(
    "Someone asked me to hang out on Thanksgiving, draft me a message informing them about my plans if I have any, if not, then let them know I can go.",
    model="gpt-5-mini",
    max_tokens=2000,
)
print(result)
```
