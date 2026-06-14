# App

## Running the dashboard

```bash
streamlit run app/dashboard.py
# or via Make:
make app
```

The dashboard starts in **demo mode** using synthetic data (32 placeholder teams). No real data files are needed to explore the interface.

## Tabs

| Tab | Status | Description |
|---|---|---|
| Champion Probabilities | Working | Bar chart of title odds from Monte-Carlo simulation |
| Match Predictor | Working | W/D/L + expected goals for any two teams |
| Simulation Explorer | Working | Re-run Monte-Carlo with adjustable N |
| More (WIP) | Placeholder | Bracket view, prediction timeline, feature importance |

## Connecting real data

Point the `DATA_PATH` constant at the top of `dashboard.py` at a real normalised results CSV (see `src/wc2026/data/loaders.py` for the expected schema) to switch from demo to live forecasts.

## Files

| File | Description |
|---|---|
| `dashboard.py` | Main Streamlit application (the one to run) |
