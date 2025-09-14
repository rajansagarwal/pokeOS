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
python msg_withvec_llm.py --context "coordinate with Rajan about an nyc trip" --ctx_k 6 --ctx_threads 3 --dbdir ~/.chat_memdb --table messages
```

To index the messages, use the following command:

```bash
python msg_withvec_llm.py --index --dbdir ~/.chat_memdb --contacts ~/.contacts_cache.txt
```


