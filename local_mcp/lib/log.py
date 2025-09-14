import os, sys

def debug(msg: str) -> None:
    if os.getenv("DEBUG", "").strip() in {"1","true","yes"}:
        print(f"[debug] {msg}", file=sys.stderr)
