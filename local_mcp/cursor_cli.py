import os
import subprocess
import threading


def send_poke_message(message):
    """Send a message via Poke API"""
    # Mock implementation for testing
    print(f"SENT TO POKE: {message}")
    return True
    
    # Real implementation (commented out for testing)
    # api_key = os.getenv('POKE_API_KEY')
    # if not api_key:
    #     return False
    # 
    # try:
    #     response = requests.post(
    #         'https://poke.com/api/v1/inbound-sms/webhook',
    #         headers={
    #             'Authorization': f'Bearer {api_key}',
    #             'Content-Type': 'application/json'
    #         },
    #         json={'message': message}
    #     )
    #     return response.status_code == 200
    # except Exception:
    #     return False

def run_command(command, cwd=None):
    """Run a command and return its output"""
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True, cwd=cwd)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return None, str(e), -1

def _monitor_cursor_completion():
    """Monitor for cursor-agent completion by checking process list"""
    import time
    import subprocess
    
    # Wait a bit for cursor-agent to start
    time.sleep(2)
    
    # Monitor for cursor-agent processes
    while True:
        try:
            # Check if cursor-agent is still running
            result = subprocess.run(
                ["pgrep", "-f", "cursor-agent"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:  # No cursor-agent processes found
                send_poke_message("Cursor agent completed! Check your changes.")
                break
                
        except Exception:
            pass
            
        time.sleep(5)  # Check every 5 seconds

def run_cursor_agent(prompt, project_dir=None):
    """Run cursor-agent with the given prompt in the specified directory"""
    if project_dir and not os.path.isdir(project_dir):
        return "Invalid directory"
    
    if not project_dir:
        project_dir = os.getcwd()
    
    command = f"cursor-agent -p \"{prompt}\""
    
    try:
        # Use osascript to open cursor-agent in a new terminal window
        escaped_command = command.replace('"', '\\"')
        applescript = f'''
        tell application "Terminal"
            do script "cd {project_dir} && {escaped_command}"
            activate
        end tell
        '''
        
        # Start the AppleScript
        subprocess.run(["osascript", "-e", applescript])
        
        # Start background monitoring thread
        monitor_thread = threading.Thread(target=_monitor_cursor_completion)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return "Cursor agent started in new terminal"
    except Exception as e:
        return f"Failed to start: {e}"

def list_files(path="."):
    """List files in a directory"""
    stdout, stderr, returncode = run_command(f"ls -la {path}")
    if returncode == 0:
        return stdout
    return None

def cat_file(file_path):
    """Read and return file contents"""
    stdout, stderr, returncode = run_command(f"cat {file_path}")
    if returncode == 0:
        return stdout
    return None

if __name__ == "__main__":
    cr = run_cursor_agent("Make a file called hello.py that prints hello world", "/Users/rajanagarwal/Downloads/Work/poke-desktop/")
    print(cr)