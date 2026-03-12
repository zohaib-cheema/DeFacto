"""Training pipelines and evaluation with an untouched external test split."""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import Dataset, validate_pair


def select_text(frame: pd.DataFrame) -> pd.Series:
    return frame.iloc[:, 0]


def build_pipeline(mode: str, model: str, seed: int) -> Pipeline:
    classifiers = {
        "logistic": LogisticRegression(max_iter=1500, random_state=seed),
        "forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=400,
            early_stopping=True,
            random_state=seed,
        ),
    }
    if model not in classifiers:
        raise ValueError("Unknown model")
    if mode == "text":
        if model != "logistic":
            raise ValueError("The text baseline supports logistic regression")
        steps = [
            ("text", FunctionTransformer(select_text, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
        ]
    else:
        steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    return Pipeline([*steps, ("classifier", classifiers[model])])


def bootstrap_accuracy(y_true, y_pred, seed: int, samples: int = 500) -> list[float]:
    """Percentile interval conditional on this fixed fitted model and test set."""
    correct = np.asarray(y_true) == np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = [rng.choice(correct, len(correct), replace=True).mean() for _ in range(samples)]
    return np.quantile(scores, [0.025, 0.975]).tolist()


def train_experiment(
    train: Dataset,
    test: Dataset,
    output: str | Path,
    model: str = "logistic",
    seed: int = 42,
    folds: int = 5,
) -> dict:
    overlap = validate_pair(train, test)
    if folds < 2 or train.labels.value_counts().min() < folds:
        raise ValueError("Each training class needs at least one row per CV fold")
    pipeline = build_pipeline(train.mode, model, seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        pipeline,
        train.features,
        train.labels,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
        error_score="raise",
    )
    pipeline.fit(train.features, train.labels)
    prediction = pipeline.predict(test.features)
    probability = pipeline.predict_proba(test.features)
    classes = pipeline.classes_.tolist()
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(train.labels), 1)), train.labels)
    dummy_pred = dummy.predict(np.zeros((len(test.labels), 1)))
    run_id = uuid4().hex[:16]
    run = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "mode": train.mode,
        "seed": seed,
        "folds": folds,
        "train": train.profile(),
        "test": test.profile(),
        "overlap": overlap,
        "classes": classes,
        "cv_macro_f1": scores.tolist(),
        "metrics": {
            "accuracy": float(accuracy_score(test.labels, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(test.labels, prediction)),
            "macro_f1": float(f1_score(test.labels, prediction, average="macro")),
            "log_loss": float(log_loss(test.labels, probability, labels=classes)),
            "majority_accuracy": float(accuracy_score(test.labels, dummy_pred)),
            "cv_macro_f1_mean": float(scores.mean()),
            "cv_macro_f1_std": float(scores.std()),
        },
        "accuracy_interval": bootstrap_accuracy(test.labels, prediction, seed),
        "classification": classification_report(
            test.labels,
            prediction,
            labels=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion": confusion_matrix(test.labels, prediction, labels=classes).tolist(),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "parameters": pipeline.get_params()["classifier"].get_params(),
    }
    destination = Path(output) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    joblib.dump({"pipeline": pipeline, "run": run}, destination / "model.joblib")
    predictions = pd.DataFrame(
        {
            "row_number": np.arange(1, len(test.labels) + 1),
            "actual": test.labels.to_numpy(),
            "predicted": prediction,
            "confidence": probability.max(axis=1),
        }
    )
    for position, label in enumerate(classes):
        predictions[f"probability_{label}"] = probability[:, position]
    predictions.to_csv(destination / "predictions.csv", index=False)
    run["artifact_directory"] = str(destination)
    return run


def predict_file(model_path: str | Path, input_path: str | Path, output_path: str | Path) -> int:
    """Only load model artifacts produced by a trusted local experiment."""
    artifact = joblib.load(model_path)
    run, pipeline = artifact["run"], artifact["pipeline"]
    frame = pd.read_csv(input_path)
    columns = run["train"]["features"]
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Missing input features: {sorted(missing)}")
    features = frame[columns].copy()
    if run["mode"] == "numeric":
        features = features.apply(pd.to_numeric, errors="raise")
        if np.isinf(features.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("Features cannot contain infinity")
    else:
        features = features.astype("string")
        if features.iloc[:, 0].isna().any() or features.iloc[:, 0].str.strip().eq("").any():
            raise ValueError("Text inputs must be nonempty")
    if frame.empty:
        raise ValueError("The input file is empty")
    result = pd.DataFrame(
        {
            "row_number": np.arange(1, len(frame) + 1),
            "predicted": pipeline.predict(features),
            "confidence": pipeline.predict_proba(features).max(axis=1),
        }
    )
    result.to_csv(output_path, index=False)
    return len(result)
