# World Cup 2026 Knockout Predictor

A production-shaped ML system that predicts the FIFA World Cup 2026 knockout stage: match win/draw/loss probabilities, expected goals, and Monte Carlo tournament simulation (Round of 32 → Champion), designed to be re-run after every completed round.

This is an **MVP**: a fully working, leakage-safe, tested pipeline covering everything that's realistically possible with the two data sources named below. It is deliberately not the full spec in one shot — see [Roadmap](#roadmap--todo) for what's next and why.

## Data sources

### 1. `mominullptr/FIFA-World-Cup-2026-Dataset` (GitHub, CC0) — wired up, schema-verified

11 relational CSVs covering the real 48-team, 12-group 2026 format (teams, venues, tournament stages, matches with xG, squads/players with market value, per-match team stats). Column names in `src/worldcup_predictor/data/schemas.py` were verified against the live files, not guessed.

### 2. `piterfm/fifa-football-world-cup` (Kaggle, 1930–2022) — downloaded, schema verified, **wired into backtesting**

Kaggle serves a JS app to unauthenticated requests, so this dataset's real columns could not be fetched during initial development. Once Kaggle credentials were added to `.env`, the real dataset was downloaded and the schema reconciled (2026-07-05). It ships 5 files; only the match-level one is integrated so far:

- `matches_1930_2022.csv` (964 rows, 1930–2022) — **loaded, schema-verified, normalized into the canonical `MatchRecord` schema** (`build_historical_match_table` in `preprocessing.py`) and fed through a reduced Elo/rolling-form feature pipeline (`build_historical_feature_frame`) into a real time-based backtest (`worldcup-predictor backtest-historical`; see Usage below).
- `world_cup.csv`, `schedule_2026.csv`, `fifa_ranking_2022-10-06.csv`, `fifa_ranking_2026-06-08.csv` — downloaded but **not yet loaded/validated** by the pipeline (see roadmap).
- Only Elo, rolling form, rest days and neutral-venue (vs. `Host`) are computed from this dataset — no squad market value or per-match FIFA ranking exist for most of 1930-2022, so the historical backtest compares a reduced model/feature set against Elo-only, Poisson-only and historical-win-rate baselines (not FIFA-ranking-only or market-value-only, which the spec also asks for but this data can't support before 2026).
- Historical stage naming includes older formats no longer used (e.g. two round-robin "group" rounds in 1950-1982) — mapped to `group_stage`/non-knockout, since they were round-robin, not sudden-death (`HISTORICAL_STAGE_MAP` in `preprocessing.py`).
- Team names differing between the two datasets are aliased (`HISTORICAL_TEAM_ALIASES`): "United States"→"USA", "Korea Republic"→"South Korea", "Czech Republic"→"Czechia". "Czechoslovakia" is deliberately **not** merged into "Czechia" — different national-team eras, not a renaming.
- `worldcup-predictor download-data --source historical` requires `KAGGLE_USERNAME` / `KAGGLE_KEY` in `.env` (see `.env.example`) and fails loudly with setup instructions if they're missing — it never silently substitutes fixture data for a real run.

### Feasibility matrix

| Capability | Status |
|---|---|
| Elo, FIFA ranking, rolling form, market value, squad age/caps, venue altitude, rest days, Poisson goal model, real bracket simulation | ✅ implemented now, wc2026 dataset |
| Historical backtesting (train-until-2014→test-2018, train-until-2018→test-2022) | ✅ implemented, `worldcup-predictor backtest-historical`, reduced Elo/form-only feature set |
| Lineup/event/referee-based features (starting-XI market value, cards) | ✅ data exists in both datasets (`match_lineups.csv`, `match_events.csv`, `referees.csv`, and the historical file's own card/sub columns) — Phase 2, not wired up |
| Other competitions (Euro, Copa América, Nations League, …) | 🧩 architecture supports it (any loader that emits `MatchRecord` plugs in) but no source is wired |
| Travel distance between venues per team | ⚠️ approximable from venue geocoordinates, not implemented |
| Injuries/suspensions, bookmaker odds | ❌ not in either dataset; odds would need a paid API |

## Architecture

```
data/{raw,interim,processed,external}     raw → schema-validated → canonical MatchRecord → features
src/worldcup_predictor/
  data/          loaders (download + load), pandera schemas, validation, raw→interim preprocessing
  features/      elo.py, rolling.py, group_stage.py, squad.py, feature_pipeline.py (orchestrator)
  models/        baselines.py, train.py (Poisson goal model + LightGBM outcome model), predict.py,
                 calibration.py (isotonic), evaluation.py (accuracy/F1/log loss/Brier/RPS + time splits)
  simulation/    match_simulator.py (Poisson scoring), knockout.py (ET + penalties, never a draw),
                 tournament.py (100k-run Monte Carlo bracket walk), probabilities.py
  reporting/     reports.py (Markdown), plots.py (Plotly, shared by dashboard)
  app/           streamlit_app.py — reads only from outputs/, never recomputes
  cli.py         worldcup-predictor <command>
```

Data flows one direction only: raw → interim (`preprocessing.py`) → features (`feature_pipeline.py`) → models → simulation → reports/dashboard. Nothing downstream mutates upstream state.

## Leakage rules (enforced, not just documented)

1. Elo and rolling-form features are computed in one **chronological left-to-right pass**: a match's pre-match rating/form is a pure function of strictly earlier matches. `tests/test_feature_leakage.py` proves this by truncating the dataset after each match and asserting identical feature values.
2. Group-stage summary features (points, GD, rank, qualification) are attached **only to knockout rows** — group-stage rows get `None` for these columns, checked by test, not by convention.
3. A match's own xG/score is never in the feature list used to predict its own outcome (`test_model_configs_never_use_the_matchs_own_xg_as_a_feature`).
4. Knockout matches can never resolve to a draw without extra time + penalties recorded — enforced in `MatchRecord`'s validator and in the simulator's `knockout.py`.
5. A real bug of exactly this class was caught during development: `pandas.DataFrame` construction silently turned `None` into float `NaN` for the `winner` column, and every `x is not None` check downstream (Elo, rolling form, the bracket resolver) was tricked into treating unplayed matches as decided draws. Fixed by switching every such check to `pd.notna(...)`; a regression test (`test_simulate_tournament_final_is_a_genuine_coin_toss_between_close_teams`) guards against it recurring.
6. Rule 4 above ("knockout can't end in a draw without penalties") turned out to be *format-specific, not universal*: real 1930s-1960s World Cup data has drawn knockout matches resolved by a separate replay instead of extra time/penalties (verified real case: Brazil 1-1 Czechoslovakia, 1938 quarter-final). `MatchRecord` no longer hard-rejects this; the "always penalties on a draw" invariant is instead checked as a business rule specific to the current dataset (`test_wc2026_knockout_matches_never_end_in_a_draw`), not baked into the shared schema. Lesson: verify an invariant against real data across the *whole* dataset before enforcing it universally, not just against a hand-written fixture.

## Setup

```bash
pip install uv          # if you don't have it
uv venv
uv pip install -e ".[dev,kaggle]"
cp .env.example .env    # optional: only needed for the real Kaggle download
```

## Usage

Every command works immediately against bundled example data — no download, no network, no credentials:

```bash
worldcup-predictor predict-matches --fixtures
worldcup-predictor simulate-tournament --fixtures
worldcup-predictor update-after-round --fixtures     # both of the above + all reports
worldcup-predictor backtest-historical --fixtures    # time-based backtest on 1930-2022 data

# Once you have real data:
worldcup-predictor download-data --source wc2026
worldcup-predictor download-data --source historical   # requires Kaggle credentials in .env
worldcup-predictor update-after-round                  # drop --fixtures
worldcup-predictor backtest-historical                 # drop --fixtures
```

Outputs land in `outputs/predictions/`, `outputs/simulations/`, `outputs/reports/` (all gitignored — regenerate, don't commit).

Dashboard:

```bash
worldcup-predictor update-after-round --fixtures   # generate outputs first
streamlit run src/worldcup_predictor/app/streamlit_app.py
```

3 of the planned 8 pages are in the MVP (Champion Probabilities, Match Predictor, Tournament Bracket) — the rest need training/backtest artifacts this MVP doesn't persist yet (see roadmap).

## Tests

```bash
pytest                    # 47 tests, all against bundled fixtures, no network
ruff check src tests
mypy src
```

`tests/test_feature_leakage.py` is the most important file in the repo — read it before changing any feature code.

## Configuration

Nothing is hardcoded: `configs/{data,features,model,simulation}.yaml`, loaded into validated Pydantic models (`src/worldcup_predictor/utils/config.py`). Change Elo K-factors, rolling-form windows, model hyperparameters, or Monte Carlo run count there, not in code.

## Roadmap / To-do (priority order)

1. **Model comparison** — add XGBoost/CatBoost/Random Forest/Logistic Regression variants next to the current LightGBM outcome model; full metric suite per model (currently: accuracy, F1, log loss, Brier, RPS — already implemented for both the 2026 predictions and the historical backtest, just needs more models to compare). The historical backtest already shows the Elo-only baseline beating both trained models on RPS in both splits — worth investigating once more models are added.
2. **MLflow tracking** — persist trained model artifacts instead of retraining from scratch per CLI invocation (fine at this data size, won't be once the real 48-team dataset with hundreds of matches is used across many re-runs).
3. **Remaining 5 dashboard pages** — Team Comparison, Prediction Timeline, Feature Importance, Simulation Explorer, Model Diagnostics. The historical backtest results are a natural data source for Model Diagnostics now.
4. **Lineup/event/referee features** — starting-XI market value (`match_lineups.csv`), card rates (`match_events.csv`, `referees.csv`, and the historical file's own card/substitution columns) — data already downloaded, just not wired into `features/squad.py` yet.
5. **Bracket topology verification** — `simulation/tournament.py` infers stage-to-stage pairing by sorted `match_id` adjacency when a future round isn't yet in the data; verify this against the real dataset's actual Round of 32→Final match ordering once fully drawn, and replace with the verified official bracket if it differs.
6. **Additional historical files** — `world_cup.csv` (per-tournament summary), `schedule_2026.csv`, and the two `fifa_ranking_*.csv` snapshots are downloaded but not loaded; the ranking snapshots in particular could extend per-match FIFA ranking features back to ~1992.
7. **Additional competitions** — any source that emits a `MatchRecord`-shaped frame plugs into the existing pipeline (Euro, Copa América, Nations League, qualifiers, friendlies). No source is wired up yet.
8. **Travel distance / in-simulation rating updates** — approximate travel km from venue geocoordinates; consider re-estimating team strength between simulated rounds instead of freezing it for the whole Monte Carlo run (current documented simplification, see `simulation/tournament.py` docstring).
