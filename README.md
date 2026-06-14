# ⚽ WC2026 Predictor

Production-grade ML system that forecasts the **FIFA World Cup 2026 knockout
stage**: per-match win/draw/loss probabilities, expected goals, full
Monte-Carlo tournament simulation, and live re-forecasting after every
knockout round.

> **Status:** the analytic core (Elo → Poisson → Monte-Carlo) is **fully
> implemented, tested, and runnable today**. Feature-rich extensions (xG,
> market values, FIFA-ranking joins, learned gradient-boosting goal model) are
> scaffolded with clean interfaces and explicit `TODO` markers — see
> [Data reality check](#data-reality-check).

---

## Quickstart

```bash
# 1. install (uv or pip; Python 3.11–3.12)
pip install -e ".[dev]"

# 2. run the synthetic end-to-end demo (no data needed)
wc2026 demo --n 5000

# 3. launch the dashboard
streamlit run app/dashboard.py

# 4. run the test suite
pytest
```

For a real forecast you supply two files and run:

```bash
wc2026 forecast \
  --history data/processed/results.csv \
  --seeding data/processed/wc2026_seeding.csv \
  --config configs/default.yaml
```

---

## How it works

```
results history ──▶ EloEngine.replay ──▶ pre-match Elo + final ratings
                                              │
                            ┌─────────────────┴───────────────┐
                            ▼                                  ▼
                  EloPoissonModel                    (optional) LearnedGoalModel
                  expected goals λ_home, λ_away      LightGBM/XGBoost on features
                            │
                            ▼
            Poisson scoreline grid  ──▶  W/D/L probs + sampled scorelines
                            │
                            ▼
              TournamentSimulator (Monte-Carlo, 100k runs over the 32-team bracket)
                            │
                            ▼
        per-team P(R16), P(QF), P(SF), P(Final), P(Champion)  ──▶ reports + dashboard
```

1. **Elo** — a dynamic international rating updated chronologically after every
   match, with tournament-importance and goal-difference weighting
   (`eloratings.net` convention). `replay()` emits leakage-free *pre-match*
   ratings for feature building and ends at current ratings to seed simulation.
2. **Goal model** — the default `EloPoissonModel` maps the Elo difference to
   expected goals for each side; goals are independent Poisson variables. A
   learned LightGBM/XGBoost regressor plugs into the same interface.
3. **Match resolution** — scoreline distribution = outer product of the two
   Poisson PMFs (truncated at `max_goals`). Knockout ties are broken by an
   Elo-tilted shootout.
4. **Tournament simulation** — `TournamentSimulator` plays the bracket `N`
   times and aggregates round-reach and title probabilities. Reproducible via a
   seeded RNG.
5. **Dynamic update** — `pipeline.update_after_round()` appends completed
   knockout results, refreshes Elo, and re-simulates the remaining bracket,
   producing fresh forecasts after the group stage, R32, R16, QF and SF.

---

## WC 2026 format note

WC 2026 uses the new **48-team** format: 12 groups of 4 → a **32-team
knockout** (Round of 32 → R16 → QF → SF → Final). The bracket topology is built
in `simulation/bracket.py`. The exact slot map (which group's winner meets
which) is flagged `TODO(bracket)` and must be set from the official FIFA
schedule before a production run.

---

## Data reality check

Several features in the original spec are **not freely or cleanly available**
for national teams. The system is built to degrade gracefully: the analytic
backend needs only **results + Elo**, both of which are solid.

| Feature group | Availability | Status in repo |
|---|---|---|
| Match results (1872–today) | ✅ Kaggle `martj42/international-football-results` | **Implemented** (`data/loaders.py`) |
| Elo ratings | ✅ Computed in-repo from results | **Implemented** (`models/elo.py`) |
| FIFA rankings | 🟡 Kaggle mirror `cashncarry/fifaworldranking` | `TODO(data)` — as-of join stub |
| xG (national teams) | 🔴 Sparse; only some tournaments | `TODO(features-xg)` — null placeholder |
| Market values | 🔴 Transfermarkt has no API; scraping breaks ToS | `TODO(features-mv)` — null placeholder |
| Injuries / availability | 🔴 No structured historical source | `TODO(features-injury)` — null placeholder |
| Squad quality (caps, league) | 🟡 Constructible with effort | `TODO(features-squad)` — null placeholder |

Diff features (`EloDiff`, `FormDiff`, `RestDayDiff`) that depend only on
available data are **fully computed**; the rest are present as typed null
columns so the learned model can absorb them later without schema changes.

---

## Project layout

```
src/wc2026/
  config/      Pydantic config schema (configs/*.yaml)
  data/        loaders + canonical schema/contracts
  features/    leakage-safe feature engineering (form, rolling, diffs)
  models/      elo engine, elo→poisson goal model, learned-model stub
  simulation/  bracket topology + Monte-Carlo simulator
  evaluation/  classification / regression / calibration metrics
  reporting/   markdown report generation
  pipeline.py  orchestration: run_forecast, update_after_round
  cli.py       Typer CLI (forecast, demo)
app/           Streamlit dashboard
configs/       default.yaml
tests/         pytest suite (core fully covered)
.github/       CI (ruff + mypy + pytest, py3.11/3.12)
```

---

## Roadmap

**Phase 0 — core (done).** Elo, Poisson goal model, bracket, Monte-Carlo,
reports, dashboard skeleton, CI, tests.

**Phase 1 — real data wiring.** Vendor the Kaggle results CSV; implement the
FIFA-ranking as-of join; define the official 2026 bracket slot map; produce the
first real post-group-stage forecast.

**Phase 2 — learned goal model.** Train LightGBM Poisson regressors on
Elo + form features with a strictly *temporal* validation split; MLflow
tracking + model registry; compare against the analytic baseline on log-loss /
Brier / champion log-loss.

**Phase 3 — calibration & evaluation.** Fit the Elo→goals constants on the
training window; reliability curves; backtest against WC 2022 and compare
champion probabilities to bookmaker-implied odds.

**Phase 4 — richer features.** Squad quality, rest/load, and (where obtainable)
xG. Each is an isolated module behind an existing null column.

**Phase 5 — dashboard completion.** Bracket view, prediction timeline, team
comparison, SHAP feature-importance tab.

---

## Prioritised to-do list

1. **`TODO(data)`** — vendor `results.csv`; implement `load_fifa_rankings` as-of join.
2. **`TODO(bracket)`** — replace sequential seeding with the official FIFA 2026 slot map.
3. **Calibration** — fit `EloPoissonModel` constants on the training window (`evaluation/calibration.py`, to add).
4. **`TODO(model-train)`** — finish `LearnedGoalModel.train` with MLflow + temporal CV.
5. **Backtest** — replay WC 2022 to validate champion log-loss vs. bookmakers.
6. **`TODO(features-*)`** — squad quality first (most tractable), then xG, then market value.
7. **Dashboard** — bracket + timeline tabs once forecast history is persisted.

---

## Design principles

- **No data leakage.** Every feature is computable strictly before kickoff;
  rolling features use `shift(1)`; Elo columns are pre-match by construction.
- **Backend-agnostic simulator.** Analytic and learned goal models share one
  `GoalModel` protocol.
- **Fail fast.** Pydantic validates config at load; the simulator asserts a
  fully-assigned bracket and checks probability invariants in tests.
- **Reproducible.** Seeded RNG, pinned deps, deterministic Elo replay.

## License

MIT.
