import subprocess, os, sys, argparse, pathlib

APPLE_SCRIPT = r'''
tell application "Contacts"
  set out to ""
  repeat with p in people
    set pname to (name of p as string)
    repeat with ph in phones of p
      set out to out & pname & "||phone||" & (value of ph as string) & linefeed
    end repeat
    repeat with em in emails of p
      set out to out & pname & "||email||" & (value of em as string) & linefeed
    end repeat
  end repeat
  return out
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
    ap.add_argument("--out", default=os.path.expanduser("~/.contacts_cache.txt"),
                    help="Output path for the contacts cache (default: ~/.contacts_cache.txt)")
    args = ap.parse_args()

    lines = dump_contacts_lines()
    out_path = pathlib.Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    phones = sum(1 for ln in lines if "||phone||" in ln)
    emails = sum(1 for ln in lines if "||email||" in ln)
    print(f"Wrote {len(lines)} lines to {out_path}")
    print(f"  phones: {phones}, emails: {emails}")

if __name__ == "__main__":
    main()
