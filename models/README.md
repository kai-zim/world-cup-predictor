# Models

Serialised model artefacts are stored here. All files except this README are git-ignored.

## Artefacts

| File | Produced by | Description |
|---|---|---|
| `goal_model.joblib` | `make train` | Fitted `LearnedGoalModel` (LightGBM Poisson regressors for home/away expected goals) |

## Loading a model

```python
from wc2026.models.learned import LearnedGoalModel

model = LearnedGoalModel.load("models/goal_model.joblib")
lam_h, lam_a = model.expected_goals(home_elo=1650, away_elo=1580, neutral=True)
```

## Versioning

Production runs should register models in MLflow (`mlruns/`) via `mlflow.log_artifact`. The joblib file here is the local copy for quick re-use without a registry query.

Until `make train` has been run against a real results CSV, the default pipeline falls back to the analytic `EloPoissonModel` — no artefact is required for `make simulate` or `make app`.
