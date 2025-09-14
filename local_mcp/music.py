import os
import subprocess
from typing import Tuple


def _run_osascript(applescript: str) -> Tuple[str, str, int]:
    try:
        proc = subprocess.run(
            ["osascript", "-e", applescript],
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.stdout.strip(), proc.stderr.strip(), proc.returncode
    except Exception as e:
        return "", str(e), 1


def play_liked_songs() -> str:
    username = os.getenv("SPOTIFY_USERNAME")
    if username:
        return "Error: SPOTIFY_USERNAME not set in environment/.env"

    track_uri = f"spotify:user:{username}:collection"
    applescript = f'tell application "Spotify" to play track "{track_uri}"'

    stdout, stderr, code = _run_osascript(applescript)
    if code != 0:
        err = f"osascript failed ({code})"
        if stderr:
            err += f": {stderr}"
        return err
    return stdout or "Playing Liked Songs in Spotify"

