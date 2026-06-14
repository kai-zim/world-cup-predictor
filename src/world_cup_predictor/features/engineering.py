import pandas as pd


def build_form_feature(df: pd.DataFrame, team_col: str = "team", goals_col: str = "goals_for") -> pd.DataFrame:
    engineered = df.copy()
    engineered["form_last_5"] = (
        engineered.groupby(team_col)[goals_col]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return engineered
