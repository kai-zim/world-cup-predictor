import pandas as pd


def summarize_probabilities(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.describe(percentiles=[0.1, 0.5, 0.9]).T
