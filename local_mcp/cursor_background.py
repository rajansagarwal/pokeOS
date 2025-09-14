#!/usr/bin/env python3

import requests
import dotenv
import os
import json
from typing import Dict, Any, Optional

dotenv.load_dotenv()

CURSOR_API_KEY = os.getenv('CURSOR_API_KEY')
API_BASE_URL = "https://api.cursor.com"

def launch_agent(
    prompt_text: str,
    source_repository: str,
    source_ref: str = "main",
    prompt_images: Optional[list] = None,
    target_branch_name: Optional[str] = None,
    auto_create_pr: bool = False,
    model: Optional[str] = None
) -> Dict[str, Any]:
    if not CURSOR_API_KEY:
        raise ValueError("CURSOR_API_KEY environment variable not set")
        
    url = f"{API_BASE_URL}/v0/agents"
    headers = {"Authorization": f"Bearer {CURSOR_API_KEY}", "Content-Type": "application/json"}
    
    prompt = {"text": prompt_text}
    if prompt_images:
        prompt["images"] = prompt_images
    
    source = {"repository": source_repository, "ref": source_ref}
    payload = {"prompt": prompt, "source": source}
    
    if model:
        payload["model"] = model
    
    if target_branch_name:
        payload["target"] = {"branchName": target_branch_name, "autoCreatePr": auto_create_pr}
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 201:
        raise Exception(f"Failed to create agent: {response.status_code} {response.text}")
        
    return response.json()


def get_agent_status(agent_id: str) -> Dict[str, Any]:
    if not CURSOR_API_KEY:
        raise ValueError("CURSOR_API_KEY environment variable not set")
        
    url = f"{API_BASE_URL}/v0/agents/{agent_id}"
    headers = {"Authorization": f"Bearer {CURSOR_API_KEY}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to get agent status: {response.status_code} {response.text}")
        
    return response.json()


def get_agent_conversation(agent_id: str) -> Dict[str, Any]:
    if not CURSOR_API_KEY:
        raise ValueError("CURSOR_API_KEY environment variable not set")
        
    url = f"{API_BASE_URL}/v0/agents/{agent_id}/conversation"
    headers = {"Authorization": f"Bearer {CURSOR_API_KEY}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to get agent conversation: {response.status_code} {response.text}")
        
    return response.json()


def add_agent_followup(agent_id: str, prompt_text: str, prompt_images: Optional[list] = None) -> Dict[str, Any]:
    if not CURSOR_API_KEY:
        raise ValueError("CURSOR_API_KEY environment variable not set")
        
    url = f"{API_BASE_URL}/v0/agents/{agent_id}/followup"
    headers = {"Authorization": f"Bearer {CURSOR_API_KEY}", "Content-Type": "application/json"}
    
    prompt = {"text": prompt_text}
    if prompt_images:
        if len(prompt_images) > 5:
            raise ValueError("Maximum of 5 images allowed in followup prompt")
        prompt["images"] = prompt_images
    
    payload = {"prompt": prompt}
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Failed to add followup: {response.status_code} {response.text}")
        
    return response.json()


def list_repositories() -> Dict[str, Any]:
    if not CURSOR_API_KEY:
        raise ValueError("CURSOR_API_KEY environment variable not set")
        
    url = f"{API_BASE_URL}/v0/repositories"
    headers = {"Authorization": f"Bearer {CURSOR_API_KEY}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to list repositories: {response.status_code} {response.text}")
        
    return response.json()


def run_agent(prompt, repo="https://github.com/rajansagarwal/observe"):
    result = launch_agent(prompt_text=prompt, source_repository=repo)
    print(f"Created agent: {result['id']}")
    return result['id']

def check_agent(agent_id):
    status = get_agent_status(agent_id)
    print(f"Status: {status['status']}")
    if 'summary' in status and status['summary']:
        print(f"Summary: {status['summary']}")
    return status

def send_followup(agent_id, prompt):
    result = add_agent_followup(agent_id=agent_id, prompt_text=prompt)
    print(f"Followup sent to: {result['id']}")
    return result

def show_conversation(agent_id, limit=-1):
    try:
        conv = get_agent_conversation(agent_id)
        print(f"Found {len(conv['messages'])} messages")
        print(conv)
        
        for i, msg in enumerate(conv['messages'], 1):
            sender = "User" if msg['type'] == "user_message" else "Agent"
            text = msg['text']
            print(f"\n{i}. {sender}: {text}")
    except Exception as e:
        print(f"Error: {e}")

def show_repos():
    try:
        repos = list_repositories()
        print(f"Found {len(repos['repositories'])} repos:")
        for repo in repos['repositories']:
            print(f"- {repo['repository']}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Use an existing agent ID or create a new one
    agent_id = "bc-2ec08a23-93db-46ca-9f9a-935255ca244d"
    
    # Uncomment the operation you want to perform
    # agent_id = run_agent("Summarize this repository")
    # check_agent(agent_id)
    # send_followup(agent_id, "Add more details about installation")
    show_conversation(agent_id)