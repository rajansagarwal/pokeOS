import asyncio
import threading
from nova_act import NovaAct
import dotenv

dotenv.load_dotenv()

def send_to_nova(task, website):
    """Run NovaAct in a separate thread to avoid asyncio conflicts"""
    def run_in_thread():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with NovaAct(starting_page=website) as nova:
                return nova.act(task)
        finally:
            loop.close()
    
    # Use threading to run NovaAct outside the main asyncio loop
    result_container = []
    exception_container = []
    
    def thread_target():
        try:
            result = run_in_thread()
            result_container.append(result)
        except Exception as e:
            exception_container.append(e)
    
    thread = threading.Thread(target=thread_target)
    thread.start()
    thread.join()  # No timeout here - let MCP server handle it
    
    if exception_container:
        raise exception_container[0]
    
    return result_container[0] if result_container else "Task completed but no result returned"
