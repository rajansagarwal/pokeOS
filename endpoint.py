#!/usr/bin/env python3
import os
from fastmcp import FastMCP
from local_mcp.computeruse import send_to_nova
from local_mcp.cursor_background import launch_agent, get_agent_status, get_agent_conversation, add_agent_followup, list_repositories, run_agent, check_agent, send_followup, show_conversation, show_repos
from local_mcp.cursor_cli import send_poke_message, run_command, run_cursor_agent, list_files, cat_file
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