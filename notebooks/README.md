# Notebooks

Exploratory notebooks live here. They are **not part of the production pipeline** — all production code lives in `src/wc2026/`.

## Naming convention

```
00_quickstart.ipynb          — end-to-end demo without real data
01_elo_calibration.ipynb     — fit and inspect EloPoissonModel constants
02_feature_exploration.ipynb — EDA on the match history feature matrix
03_model_comparison.ipynb    — analytic vs learned model on log-loss / Brier
04_backtest_wc2022.ipynb     — replay WC 2022 and compare to bookmaker odds
```

## Guidelines

- Never import from test files or call pipeline side-effects in notebooks.
- Keep notebooks clean: clear all outputs before committing (`Kernel → Restart & Clear Output`).
- Notebook outputs are git-ignored (`.ipynb_checkpoints/`).
- Prefer importing from `wc2026.*` rather than duplicating logic inline.
