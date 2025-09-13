from datetime import datetime, timedelta, timezone
import osxphotos as oxp
from osxphotos import QueryOptions

X = 7  # past X days
now = datetime.now(timezone.utc)
start = now - timedelta(days=X)

db = oxp.PhotosDB()

opts_taken = QueryOptions(label=["terminal"])
photos_taken = db.query(opts_taken)

print("Taken:", len(photos_taken))
for p in photos_taken[:5]:
    print(p.uuid, p.original_filename, p.date)