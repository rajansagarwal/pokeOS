from __future__ import annotations
import datetime
from typing import Optional

LOCAL_TZ    = datetime.datetime.now().astimezone().tzinfo
APPLE_EPOCH = datetime.datetime(2001, 1, 1, tzinfo=datetime.timezone.utc)

def apple_time_to_dt(raw: Optional[int|float]) -> Optional[datetime.datetime]:
    if raw is None: return None
    try: val = int(raw)
    except: return None
    seconds = val / 1_000_000_000 if abs(val) > 10**12 else val
    return (APPLE_EPOCH + datetime.timedelta(seconds=seconds)).astimezone(LOCAL_TZ)

def dt_to_iso(dt: Optional[datetime.datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None
