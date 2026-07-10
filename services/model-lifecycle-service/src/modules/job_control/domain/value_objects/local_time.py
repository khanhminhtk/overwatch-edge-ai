from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


HCM_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


def local_now() -> datetime:
    return datetime.now(HCM_TIMEZONE).replace(tzinfo=None)
