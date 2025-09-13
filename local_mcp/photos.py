from pathlib import Path
import shutil
import osxphotos as oxp
from osxphotos import QueryOptions

# query Photos for items labeled "Dog"
db = oxp.PhotosDB()
# opts = QueryOptions(label=["Dog"])
opts = QueryOptions(label=["whiteboard"])
photos = db.query(opts)

# destination folder
outdir = Path.home() / "Desktop" / "_tmp_photos"
outdir.mkdir(parents=True, exist_ok=True)

def safe_copy(src: Path, dest_dir: Path) -> Path:
    """Copy src to dest_dir without overwriting; returns final path."""
    dest = dest_dir / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
        return dest

    stem, suffix = src.stem, src.suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        i += 1

for p in photos:
    src_path = p.path  # original file path (may be None or not downloaded)
    if src_path:
        sp = Path(src_path)
        if sp.exists():
            final = safe_copy(sp, outdir)
            print(f"copied: {p.uuid} -> {final}")
        else:
            print(f"skip (not on disk): {p.uuid} {p.original_filename}")
    else:
        print(f"skip (no path): {p.uuid} {p.original_filename}")
