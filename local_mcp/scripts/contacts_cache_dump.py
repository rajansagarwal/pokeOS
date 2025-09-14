import subprocess, os, sys, argparse, pathlib

APPLE_SCRIPT = r'''
tell application "Contacts"
  set contactList to {}
  repeat with p in people
    set pname to (name of p as string)
    repeat with ph in phones of p
      set end of contactList to pname & "||phone||" & (value of ph as string)
    end repeat
    repeat with em in emails of p
      set end of contactList to pname & "||email||" & (value of em as string)
    end repeat
  end repeat
  set AppleScript's text item delimiters to "\n"
  set result to contactList as string
  set AppleScript's text item delimiters to ""
  return result
end tell
'''

def dump_contacts_lines(timeout_sec: int = 60) -> list[str]:
    p = subprocess.run(
        ["/usr/bin/osascript", "-e", APPLE_SCRIPT],
        capture_output=True, text=True, timeout=timeout_sec
    )
    if p.returncode != 0:
        raise SystemExit(f"osascript failed ({p.returncode}): {p.stderr.strip()}")
    return [ln for ln in (p.stdout or "").splitlines() if ln.strip()]

def main():
    ap = argparse.ArgumentParser(description="Dump Apple Contacts to text cache (Name||phone||... / Name||email||...)")
    script_dir = pathlib.Path(__file__).parent
    default_cache = script_dir / ".contacts_cache.txt"
    ap.add_argument("--out", default=str(default_cache),
                    help=f"Output path for the contacts cache (default: {default_cache})")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Timeout in seconds for AppleScript execution (default: 600)")
    args = ap.parse_args()

    lines = dump_contacts_lines(timeout_sec=args.timeout)
    out_path = pathlib.Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    phones = sum(1 for ln in lines if "||phone||" in ln)
    emails = sum(1 for ln in lines if "||email||" in ln)
    print(f"Wrote {len(lines)} lines to {out_path}")
    print(f"  phones: {phones}, emails: {emails}")

if __name__ == "__main__":
    main()
