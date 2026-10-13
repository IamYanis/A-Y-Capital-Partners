"""JSON event logs; API credentials and HTTP exception URLs are never logged."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

LOGGER = logging.getLogger("aycapital.cross_asset")


def event(level: int, name: str, **fields: Any) -> None:
    LOGGER.log(
        level,
        json.dumps(
            {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": name, **fields},
            default=str,
        ),
    )
