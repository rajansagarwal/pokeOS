import re

def is_question(task: str) -> bool:
    t = task.strip()
    if "?" in t: return True
    return bool(re.match(r"^(?i)(is|are|am|do|does|did|can|could|should|will|would|when|where|who|whom|whose|which|what|why|how)\b", t))
