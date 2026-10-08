# Alpha Lab

[![CI](https://github.com/RafaelPF07/alpha-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/RafaelPF07/alpha-lab/actions/workflows/ci.yml)

A research harness for daily equity trading signals. The goal is not to find alpha. The goal is an honest, well-tested pipeline that resists lookahead bias, overfitting and data snooping.

**Status: Phase 0 (deterministic core, no LLMs).** Later phases add agents, validation, the holdout, and a dashboard.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run lab data fetch          # downloads prices into data/ (gitignored)
uv run lab data summary        # data-quality report
uv run lab backtest momentum_12_1 --split train
```

## Anti-overfitting design (so far)

- **Lookahead tests.** Every signal must pass a truncation test and a perturbation test before it is trusted. `cheater`, a deliberately broken signal that reads tomorrow's return, is required to fail them.
- **Timing.** Weights decided at the close of day t earn the return from close t to close t+1 (`weights.shift(1) * returns`).
- **Windows.** Train, validation and holdout windows do not overlap. A return counts toward the window of the day it is realised on.
- **Holdout locking.** Holdout prices are written to a separate Parquet file. The default loader never reads it, and only `pipeline/holdout.py` may call the holdout reader. This is a convention guard, not a security boundary.
- **Costs.** Costs are charged on the change in position, at a fixed number of basis points per side.

## Known limitations

- **Survivorship bias.** The universe is today's large caps plus sector ETFs. Companies that failed or left the index before today are absent, so results are optimistic.
- **Data quality.** yfinance is a free source. Spinoffs, splits and dividend adjustments can be wrong. Daily moves beyond ±15% are flagged in `lab data summary` for manual inspection. Moves beyond ±50% are marked as extreme.
- **Turnover ignores drift.** Turnover is measured from changes in target weights. A position that drifts with prices and is rebalanced back is therefore not charged for the rebalance trades. Constant-weight signals show zero turnover and zero cost.
- **No borrowing costs for shorts, no slippage beyond fixed bps,** and daily close-to-close execution is optimistic.
- **Small universe** of 50 names.
- **Possible LLM lookahead (Phase 1+).** Language models may have learned about the post-2010 market during training. This is a subtle form of lookahead and cannot be removed by the harness alone.

## Results

To be filled in by the author after the holdout run.
