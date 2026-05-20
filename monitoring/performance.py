"""Rolling P&L + equity snapshots persisted to parquet for the dashboard.

Each `record()` appends one snapshot. The main loop is the sole writer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class PerformanceTracker:
    def __init__(self, path: str | Path = "logs/performance.parquet") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, snapshot: dict) -> None:
        snapshot = {"ts": datetime.now(timezone.utc), **snapshot}
        df = pd.DataFrame([snapshot])
        if self.path.exists():
            df = pd.concat([pd.read_parquet(self.path), df], ignore_index=True)
        df.to_parquet(self.path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.path)
