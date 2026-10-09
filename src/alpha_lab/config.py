"""Typed, validated configuration. Bad config fails at startup, not mid-run."""

from __future__ import annotations

import os
import re
import tomllib
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Dates(_Strict):
    start: date
    train_end: date
    validation_end: date
    holdout_end: date

    @model_validator(mode="after")
    def _ordered(self) -> Dates:
        if not (self.start < self.train_end < self.validation_end < self.holdout_end):
            raise ValueError("dates must satisfy start < train_end < validation_end < holdout_end")
        return self


class DataConfig(_Strict):
    ffill_limit_days: int = Field(ge=0)
    max_missing_fraction: float = Field(ge=0, le=1)
    flag_return_large: float = Field(gt=0)
    flag_return_extreme: float = Field(gt=0)

    @model_validator(mode="after")
    def _flags_ordered(self) -> DataConfig:
        if self.flag_return_large >= self.flag_return_extreme:
            raise ValueError("flag_return_large must be below flag_return_extreme")
        return self


class BacktestConfig(_Strict):
    cost_bps_per_side: float = Field(ge=0)
    max_gross_leverage: float = Field(gt=0)
    annualization_days: int = Field(gt=0)
    min_history_days: int = Field(ge=0)


class GatesConfig(_Strict):
    min_train_sharpe: float
    min_validation_sharpe: float
    max_turnover_daily: float = Field(ge=0)


class AgentsConfig(_Strict):
    ideas_model: str
    coder_model: str
    critic_model: str
    max_revisions: int = Field(ge=0)
    max_tokens_per_call: int = Field(gt=0)
    context_limit_tokens: int = Field(gt=0)


class RunConfig(_Strict):
    default_budget_usd: float = Field(gt=0)
    max_concurrency: int = Field(gt=0)
    seed: int


class DedupConfig(_Strict):
    similarity_threshold: float = Field(gt=0, le=1)


class DiversityConfig(_Strict):
    themes: list[str] = Field(min_length=1)


class SandboxConfig(_Strict):
    timeout_seconds: int = Field(gt=0)
    allowed_imports: list[str]


class ModelPricing(_Strict):
    input: float = Field(ge=0)
    output: float = Field(ge=0)
    input_over_100k: float | None = Field(default=None, ge=0)
    output_over_100k: float | None = Field(default=None, ge=0)
    cache_read: float = Field(ge=0)


class PricingConfig(_Strict):
    source: str
    verified_on: date
    long_context_threshold_tokens: int = Field(gt=0)
    models: dict[str, ModelPricing]


class Settings(_Strict):
    dates: Dates
    data: DataConfig
    backtest: BacktestConfig
    gates: GatesConfig
    agents: AgentsConfig
    run: RunConfig
    sandbox: SandboxConfig
    dedup: DedupConfig
    diversity: DiversityConfig
    pricing: PricingConfig


class Universe(_Strict):
    selection_rule: str
    sector_etfs: list[str]
    stocks: list[str]

    @property
    def tickers(self) -> list[str]:
        return [*self.sector_etfs, *self.stocks]

    @model_validator(mode="after")
    def _unique(self) -> Universe:
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("universe contains duplicate tickers")
        return self


_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ. Existing variables win.

    Accepts a UTF-8 byte-order mark (Windows PowerShell writes one), an `export` prefix, quoted
    values, and a ` # comment` after an unquoted value. A line that does not parse raises
    ValueError with its line number. The line's value is never put in the message.
    """
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8-sig")
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not _ENV_KEY.match(key):
            raise ValueError(f"{path.name} line {number}: expected KEY=VALUE")
        value = value.strip()
        if value[:1] in ("'", '"'):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0:
                raise ValueError(f"{path.name} line {number}: unterminated {quote} quote")
            rest = value[end + 1 :].strip()
            if rest and not rest.startswith("#"):
                raise ValueError(f"{path.name} line {number}: text after the closing quote")
            value = value[1:end]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
            if "'" in value or '"' in value:
                raise ValueError(f"{path.name} line {number}: stray quote in an unquoted value")
        os.environ.setdefault(key, value)


def load_settings(path: Path = CONFIG_DIR / "settings.toml") -> Settings:
    with path.open("rb") as f:
        return Settings.model_validate(tomllib.load(f))


def load_universe(path: Path = CONFIG_DIR / "universe.toml") -> Universe:
    with path.open("rb") as f:
        return Universe.model_validate(tomllib.load(f)["universe"])
