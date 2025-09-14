# poke-desktop

Local MCP server providing Poke with 5 core capabilities for enhanced digital interaction through iMessage intelligence, web automation, development tools, system control, and music integration.

## Core Capabilities

### 1. iMessage Intelligence
Proactive message analysis with vector search and LLM interpretation. Indexes messages into LanceDB with OpenAI embeddings for semantic search, context retrieval, and automated monitoring of unreplied conversations.

### 2. Web Agent Integration
Automated web interaction through Nova agents for form filling and meeting scheduling. Processes emails and executes actionable items using local browser automation instead of relying on web search alone.

### 3. Cursor IDE Integration
Direct code debugging and development assistance within your IDE. Launches background agents, manages repositories, and provides conversational interface for development tasks.

### 4. System Command Execution
Generalized command runner with intelligent AppleScript integration for macOS automation. Handles terminal commands, system settings, wellness automation, and file operations.

## Setup

### Prerequisites
Python 3.13+, uv package manager, Tailscale, and macOS Full Disk Access permissions.

### Installation

```bash
# Install dependencies
uv sync

# Setup contacts cache for name resolution (optional)
python local_mcp/scripts/contacts_cache_dump.py

# Grant permissions first (see Permissions section below)

# Initial message indexing  
cd local_mcp
python messages.py --index --chats 20 --per 30 --dbdir ~/.chat_memdb
```

This creates:
- Contacts cache at `~/.contacts_cache.txt` 
- Message snapshots in `~/Library/Messages/_snapshots/`
- Vector database at `~/.chat_memdb`

### Running the MCP Server

```bash
# Start the local MCP server
uv run endpoint.py

# In another terminal, create secure public tunnel
tailscale funnel --https=443 --set-path=/mcp "localhost:8000/mcp"
```

The server will be available locally at `http://localhost:8000/mcp` and publicly via your Tailscale funnel URL.

### Environment Variables

Create a `.env` file with:
```bash
OPENAI_API_KEY=your_openai_api_key
POKE_API_KEY=your_poke_api_key
WORKING_DIRECTORY=/path/to/your/projects
```

### Permissions

**Required for iMessage database access:**

1. **Full Disk Access**: System Settings > Security & Privacy > Privacy > Full Disk Access
   - Add Terminal.app, your IDE (VS Code, etc.), and Python executable
2. **Contacts Access**: For name resolution instead of phone numbers/emails
   - Add same applications to Contacts privacy settings

**iMessage Database Location**: `~/Library/Messages/chat.db`
- Must be readable by your terminal/Python process
- Without permissions, indexing will fail with access denied errors

## MCP Tools Available

The server exposes the following tools to Poke via MCP:

**Message Intelligence**: `retrieve_messages_text`, `suggest_message_context`, `get_relevant_context`, `revalidate_message_index`

**System & Web**: `web_agent`, `run_shell_command`, `play_spotify_liked_songs`

**Development**: `cursor_launch_agent`, `cursor_get_agent_status`, `list_directory_files`, `read_file_contents`

**Proactive**: `proactive_message_check` for automated unreplied message scanning

## Architecture

The system uses a local MCP server funneled through Tailscale SSL to provide secure remote access. This approach enables Poke to access local system capabilities while maintaining security through Tailscale's mesh networking.

Key components: LanceDB vector store, FastMCP server, Tailscale SSL tunnel, custom memory system, and cron-based proactive monitoring.

This setup was instrumental in helping Poke debug its own Spotify integration locally, demonstrating the meta-capabilities of the system.

## Automation

```bash
# Reindex messages
cd local_mcp && python messages.py --index

# Proactive check (add to crontab)
python -c "from messages import proactive_message_check; proactive_message_check()"
```
