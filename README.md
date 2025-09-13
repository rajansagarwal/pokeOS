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

The script will output a JSON file containing the recent messages for each chat.

