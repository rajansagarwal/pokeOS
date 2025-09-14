#!/usr/bin/env python3
import os
from typing import Optional
from fastmcp import FastMCP
from local_mcp.computeruse import send_to_nova
from local_mcp.cursor_background import launch_agent, get_agent_status, get_agent_conversation, add_agent_followup, list_repositories, run_agent, check_agent, send_followup, show_conversation, show_repos
from local_mcp.music import play_liked_songs as _play_liked_songs
from local_mcp.cursor_cli import send_poke_message, run_command, run_cursor_agent, list_files, cat_file
from local_mcp.messages import retrieve_text, suggest_next_message_from_prompt, _get_store
import dotenv

dotenv.load_dotenv()

mcp = FastMCP("Sample MCP Server")

@mcp.tool(description="Greet a user by name with a welcome message from the MCP server")
def greet(name: str) -> str:
    return f"Hello, {name}! Welcome to our sample MCP server running on Heroku!"

@mcp.tool(description="Get information about the MCP server including name, version, environment, and Python version")
def get_server_info() -> dict:
    return {
        "server_name": "Sample MCP Server",
        "version": "1.0.0",
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "python_version": os.sys.version.split()[0]
    }

@mcp.tool(description="Send a task to a web agent to find information. Use this for harder tasks that require live scraping, not regular web search")
def web_agent(task: str, website: str) -> str:
    result = send_to_nova(task, website)
    return result.response

@mcp.tool(description="Launch a Cursor agent with a prompt and repository")
def cursor_launch_agent(prompt_text: str, source_repository: str, source_ref: str = "main", target_branch_name: str = None, auto_create_pr: bool = False, model: str = None) -> dict:
    return launch_agent(prompt_text, source_repository, source_ref, None, target_branch_name, auto_create_pr, model)

@mcp.tool(description="Get the status of a Cursor agent")
def cursor_get_agent_status(agent_id: str) -> dict:
    return get_agent_status(agent_id)

@mcp.tool(description="Get the conversation history of a Cursor agent")
def cursor_get_agent_conversation(agent_id: str) -> dict:
    return get_agent_conversation(agent_id)

@mcp.tool(description="Add a followup message to a Cursor agent")
def cursor_add_followup(agent_id: str, prompt_text: str) -> dict:
    return add_agent_followup(agent_id, prompt_text)

@mcp.tool(description="List all repositories available in Cursor")
def cursor_list_repositories() -> dict:
    return list_repositories()

@mcp.tool(description="Run a Cursor agent with a simple prompt (returns agent ID)")
def cursor_run_agent(prompt: str, repo: str = "https://github.com/rajansagarwal/observe") -> str:
    return run_agent(prompt, repo)

@mcp.tool(description="Check the status of a Cursor agent and print summary")
def cursor_check_agent(agent_id: str) -> dict:
    return check_agent(agent_id)

@mcp.tool(description="Send a followup message to a Cursor agent")
def cursor_send_followup(agent_id: str, prompt: str) -> dict:
    return send_followup(agent_id, prompt)

@mcp.tool(description="Show conversation history for a Cursor agent")
def cursor_show_conversation(agent_id: str) -> str:
    show_conversation(agent_id)
    return "Conversation displayed"

@mcp.tool(description="Show available repositories in Cursor")
def cursor_show_repos() -> str:
    show_repos()
    return "Repositories displayed"

@mcp.tool(description="Send a message via Poke API")
def poke_send_message(message: str) -> bool:
    return send_poke_message(message)

@mcp.tool(description="Run a shell command and return output")
def run_shell_command(command: str, cwd: str = None) -> dict:
    stdout, stderr, returncode = run_command(command, cwd)
    return {"stdout": stdout, "stderr": stderr, "returncode": returncode}

@mcp.tool(description="Run cursor-agent with a prompt in a specified directory")
def run_cursor_cli_agent(prompt: str, project_dir: str = None) -> str:
    return run_cursor_agent(prompt, project_dir)

@mcp.tool(description="List files in a directory")
def list_directory_files(path: str = ".") -> str:
    return list_files(path)

@mcp.tool(description="Read and return file contents")
def read_file_contents(file_path: str) -> str:
    return cat_file(file_path)

@mcp.tool(description="List files in the main project directory")
def list_project_files() -> str:
    WORKING_DIRECTORY = os.getenv("WORKING_DIRECTORY")
    return list_files(WORKING_DIRECTORY)

@mcp.tool(description="Open Spotify and play Liked Songs for SPOTIFY_USERNAME")
def play_spotify_liked_songs() -> str:
    return _play_liked_songs()

@mcp.tool(description="Retrieve messages using text search functionality. Returns messages grouped by chat that contain the query text.")
def retrieve_messages_text(
    query: str,
    limit: int = 20,
    chat: Optional[str] = None,
    sender: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    window: int = 0
) -> dict:
    """
    Retrieve messages using text search functionality.
    
    Args:
        query: The text to search for
        limit: Maximum number of results (default: 20)
        chat: Filter by specific chat name (optional)
        sender: Filter by specific sender (optional)
        since: ISO timestamp for lower bound (optional)
        until: ISO timestamp for upper bound (optional)
        window: Include N messages before/after each hit (default: 0)
    """
    try:
        results = retrieve_text(
            query=query,
            limit=limit,
            chat=chat,
            sender=sender,
            since=since,
            until=until,
            window=window
        )
        return results
    except Exception as e:
        return {"error": f"Error retrieving messages: {str(e)}"}

@mcp.tool(description="Suggest next message from prompt using context from previous conversations.")
def suggest_message_context(
    user_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 220
) -> dict:
    """
    Suggest next message from prompt using context from previous conversations.
    
    Args:
        user_prompt: The prompt to generate a response for
        model: LLM model to use (default: gpt-4o-mini)
        temperature: Generation temperature (default: 0.2)
        max_tokens: Maximum tokens in response (default: 220)
    """
    try:
        result = suggest_next_message_from_prompt(
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return result
    except Exception as e:
        return {"error": f"Error generating message suggestion: {str(e)}"}

@mcp.tool(description="Pull relevant context (grouped by chat) for a given prompt using semantic search.")
def get_relevant_context(
    prompt: str,
    k_per_thread: int = 6,
    max_threads: int = 3
) -> dict:
    """
    Pull relevant context (grouped by chat) for a given prompt.
    
    Args:
        prompt: The prompt to find relevant context for
        k_per_thread: Messages per thread to return (default: 6)
        max_threads: Maximum threads to return (default: 3)
    """
    try:
        store = _get_store()
        context = store.relevant_context(
            prompt=prompt,
            k_per_thread=k_per_thread,
            max_threads=max_threads
        )
        return {"context": context}
    except Exception as e:
        return {"error": f"Error retrieving context: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"Starting FastMCP server on {host}:{port}")
    
    mcp.run(
        transport="http",
        host=host,
        port=port,
        path="/mcp"
    )