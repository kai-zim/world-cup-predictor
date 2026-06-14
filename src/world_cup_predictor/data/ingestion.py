from pathlib import Path

import pandas as pd


def load_matches(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
