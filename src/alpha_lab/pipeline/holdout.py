"""The only module allowed to read holdout-period data.

Phase 0 provides the guarded read only. Phase 4 adds the human-run command, its
confirmation prompt, and the ledger record that stops a second run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.config import DATA_DIR, Settings
from alpha_lab.data import loader


def holdout_slice(settings: Settings, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Holdout-period prices only (validation_end + 1 day through holdout_end)."""
    return loader.read_holdout_cache(data_dir).loc[: pd.Timestamp(settings.dates.holdout_end)]


def run_holdout(*args: object, **kwargs: object) -> None:
    raise NotImplementedError("the holdout run is implemented in Phase 4")
