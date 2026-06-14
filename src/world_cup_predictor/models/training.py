from dataclasses import dataclass

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainedModel:
    pipeline: Pipeline
    feature_columns: list[str]


def train_baseline_model(df: pd.DataFrame, target_col: str = "home_win") -> TrainedModel:
    feature_columns = [c for c in df.columns if c != target_col]
    X = df[feature_columns]
    y = df[target_col]

    numeric_preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer([("num", numeric_preprocessor, feature_columns)])

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    model.fit(X, y)
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_param("features", len(feature_columns))

    return TrainedModel(pipeline=model, feature_columns=feature_columns)
