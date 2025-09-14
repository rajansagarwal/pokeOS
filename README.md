# poke-desktop

### Messages

The `local_mcp` directory contains a Python script that reads recent messages from the Messages database and converts them into a JSON format that can be used by the MCP. The script uses a contacts cache to resolve phone numbers and email addresses to names.

The script is based on the `contacts_cache_dump.py` script located in the local_mcp/lib directory. You must first run the following to store a local copy of your contacts on your filesystem.

```bash
python3 local_mcp/lib/contacts_cache_dump.py --out ~/.contacts_cache.txt
```

To run the script, use the following command:

```bash
python local_mcp/messages.py
```

The script will output a JSON file containing the recent messages for each chat (this will output a lot of data, so you may want to limit the number of chats and messages per chat).

Use the args to control the data you want to receive. Example use cases are:

```bash
python messages.py --text "yo" --limit 5
```

```bash
python messages.py --context "coordinate with Rajan about an nyc trip" --ctx_k 6 --ctx_threads 3 --dbdir ~/.chat_memdb --table messages
```

Context: uses the prompt to fetch relevant threads of texts so like a chain of messages/multiple messages
```bash
python messages.py --context "coordinate with Rajan about an nyc trip" --ctx_threads 2 --ctx_k 8
```

Query: uses the prompt to fetch relevant messages of texts so like a single message
```bash
python messages.py --query "nyc trip with rajan" --k 5 --pretty
```

To index the messages, use the following command:

```bash
python messages.py --index --dbdir ~/.chat_memdb --contacts ~/.contacts_cache.txt
```

Note that this creates a new snapshot of the chat.db iMessage database and stores it in the ~/.chat_memdb directory. 


