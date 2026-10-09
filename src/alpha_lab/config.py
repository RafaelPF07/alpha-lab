"""Typed, validated configuration, checked when the settings file loads.

Checked here: field types and bounds (NaN and infinity are refused for every float), the order of
the dates, the flag thresholds, and that the training and validation windows can hold a score
(Settings._windows_can_hold_a_score). Problems that depend on the price data, such as a gap or a
ticker with too many missing rows, are not visible here. They appear as data errors at run time.
"""

from __future__ import annotations

import os
import re
import tomllib
from datetime import date, timedelta
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
    max_missing_fraction: float = Field(allow_inf_nan=False, ge=0, le=1)
    flag_return_large: float = Field(allow_inf_nan=False, gt=0)
    flag_return_extreme: float = Field(allow_inf_nan=False, gt=0)

    @model_validator(mode="after")
    def _flags_ordered(self) -> DataConfig:
        if self.flag_return_large >= self.flag_return_extreme:
            raise ValueError("flag_return_large must be below flag_return_extreme")
        return self


class BacktestConfig(_Strict):
    cost_bps_per_side: float = Field(allow_inf_nan=False, ge=0)
    max_gross_leverage: float = Field(allow_inf_nan=False, gt=0)
    annualization_days: int = Field(gt=0)
    min_history_days: int = Field(ge=0)


class GatesConfig(_Strict):
    min_train_sharpe: float = Field(allow_inf_nan=False)
    min_validation_sharpe: float = Field(allow_inf_nan=False)
    max_turnover_daily: float = Field(allow_inf_nan=False, ge=0)


class AgentsConfig(_Strict):
    ideas_model: str
    coder_model: str
    critic_model: str
    max_revisions: int = Field(ge=0)
    max_tokens_per_call: int = Field(gt=0)
    context_limit_tokens: int = Field(gt=0)


class RunConfig(_Strict):
    default_budget_usd: float = Field(allow_inf_nan=False, gt=0)
    max_concurrency: int = Field(gt=0)
    seed: int


class DedupConfig(_Strict):
    similarity_threshold: float = Field(allow_inf_nan=False, gt=0, le=1)


class DiversityConfig(_Strict):
    themes: list[str] = Field(min_length=1)


class SandboxConfig(_Strict):
    timeout_seconds: int = Field(gt=0)
    allowed_imports: list[str]


class ModelPricing(_Strict):
    input: float = Field(allow_inf_nan=False, ge=0)
    output: float = Field(allow_inf_nan=False, ge=0)
    input_over_100k: float | None = Field(allow_inf_nan=False, default=None, ge=0)
    output_over_100k: float | None = Field(allow_inf_nan=False, default=None, ge=0)
    cache_read: float = Field(allow_inf_nan=False, ge=0)
    # Cache writes and long-context cache reads. Unset means the LLM layer has no price for them yet.
    cache_write_5m: float | None = Field(allow_inf_nan=False, default=None, ge=0)
    cache_write_5m_over_100k: float | None = Field(allow_inf_nan=False, default=None, ge=0)
    cache_write_1h: float | None = Field(allow_inf_nan=False, default=None, ge=0)
    cache_read_over_100k: float | None = Field(allow_inf_nan=False, default=None, ge=0)

    @model_validator(mode="after")
    def _long_context_prices_come_in_pairs(self) -> ModelPricing:
        if (self.input_over_100k is None) != (self.output_over_100k is None):
            raise ValueError("input_over_100k and output_over_100k must be set together")
        return self


class PricingConfig(_Strict):
    source: str
    verified_on: date
    long_context_threshold_tokens: int = Field(gt=0)
    models: dict[str, ModelPricing]


# Weekdays a window needs beyond the history it skips. This is a margin chosen for safety, not a
# bound derived from the engine. Weekdays overcount trading days, so the margin is a rough allowance.
SCORE_MARGIN_WEEKDAYS = 21


def _weekdays(first: date, last: date) -> int:
    """Weekdays from first to last, both included. Zero when last is before first."""
    days = (last - first).days + 1
    return sum(1 for i in range(max(days, 0)) if (first + timedelta(days=i)).weekday() < 5)


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

    @model_validator(mode="after")
    def _every_agent_model_is_priced(self) -> Settings:
        """A model the agents call must have a price, or the budget and the cost ledger are wrong."""
        roles = {
            "ideas": self.agents.ideas_model,
            "coder": self.agents.coder_model,
            "critic": self.agents.critic_model,
        }
        for role, model in roles.items():
            if model not in self.pricing.models:
                raise ValueError(f"the {role} model {model!r} has no price in [pricing.models]")
        return self

    @model_validator(mode="after")
    def _windows_can_hold_a_score(self) -> Settings:
        """Refuse at startup a window that cannot hold a scored day (R3-STAT-06).

        The engine skips the first min_history_days rows and needs at least two scored rows, so a
        short train window fails every train backtest at run time. Validation needs only the margin.
        """
        d = self.dates
        train = _weekdays(d.start, d.train_end)
        if train <= self.backtest.min_history_days + SCORE_MARGIN_WEEKDAYS:
            raise ValueError(
                f"the training window ({d.start} to {d.train_end}) has about {train} weekdays, "
                f"not enough beyond min_history_days={self.backtest.min_history_days}"
            )
        validation = _weekdays(d.train_end + timedelta(days=1), d.validation_end)
        if validation <= SCORE_MARGIN_WEEKDAYS:
            raise ValueError(
                f"the validation window ({d.train_end} to {d.validation_end}) has about "
                f"{validation} weekdays, too few to score"
            )
        return self


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
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(f"{path.name} is not UTF-8 text; save it as UTF-8") from None
    seen: set[str] = set()
    pairs: list[
        tuple[str, str]
    ] = []  # parsed in full before anything is set, so a bad file sets nothing
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
        if key in seen:
            # Two lines for one key would otherwise keep the first value silently (R3-SEC-14).
            raise ValueError(f"{path.name} line {number}: {key} is set twice in this file")
        seen.add(key)
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
        pairs.append((key, value))
    for key, value in pairs:
        os.environ.setdefault(key, value)


def load_settings(path: Path = CONFIG_DIR / "settings.toml") -> Settings:
    with path.open("rb") as f:
        return Settings.model_validate(tomllib.load(f))


def load_universe(path: Path = CONFIG_DIR / "universe.toml") -> Universe:
    with path.open("rb") as f:
        return Universe.model_validate(tomllib.load(f)["universe"])
