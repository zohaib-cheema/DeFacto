"""Validate inputs without learning anything from the evaluation split."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    path: Path
    frame: pd.DataFrame
    features: pd.DataFrame
    labels: pd.Series
    target: str
    mode: str
    fingerprint: str
    excluded: list[str]

    def profile(self) -> dict:
        return {
            "file": self.path.name,
            "sha256": self.fingerprint,
            "rows": len(self.frame),
            "features": list(self.features.columns),
            "mode": self.mode,
            "target": self.target,
            "excluded_columns": self.excluded,
            "class_counts": self.labels.value_counts().sort_index().to_dict(),
            "missing_by_feature": self.features.isna().sum().astype(int).to_dict(),
            "constant_features": [c for c in self.features if self.features[c].nunique() <= 1],
            "duplicate_feature_rows": int(self.features.duplicated().sum()),
            "limitations": [
                "Prepared feature tables do not retain original statement IDs.",
                "Matching feature rows alone cannot establish duplicate statements.",
            ]
            if self.mode == "numeric"
            else [],
        }


def load_dataset(
    path: str | Path,
    target: str = "Label",
    mode: str = "numeric",
    text_column: str = "Statement",
    exclude: tuple[str, ...] = ("Search",),
) -> Dataset:
    path = Path(path)
    if mode not in {"numeric", "text"}:
        raise ValueError("Mode must be numeric or text")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        headers = next(csv.reader(stream), [])
    if (
        not headers
        or any(not name.strip() for name in headers)
        or len(set(headers)) != len(headers)
    ):
        raise ValueError("CSV column names must be nonempty and unique")
    # A string target preserves labels such as 0 and 1 across all CSV files.
    frame = pd.read_csv(path, dtype={target: "string"})
    if frame.empty:
        raise ValueError("The dataset is empty")
    if target not in frame:
        raise ValueError(f"Missing target column: {target}")
    labels = frame[target].str.strip()
    if labels.isna().any() or labels.eq("").any():
        raise ValueError("Every row needs a nonempty target label")
    labels = labels.astype(str)
    if labels.nunique() < 2:
        raise ValueError("At least two target classes are required")
    excluded = [c for c in exclude if c in frame and c != target]
    if mode == "text":
        if text_column == target:
            raise ValueError("The text column cannot also be the target")
        if text_column not in frame:
            raise ValueError(f"Missing text column: {text_column}")
        features = frame[[text_column]].astype("string")
        if features[text_column].isna().any() or features[text_column].str.strip().eq("").any():
            raise ValueError("Text inputs must be nonempty")
    else:
        features = frame.drop(columns=[target, *excluded]).copy()
        if features.shape[1] == 0:
            raise ValueError("No feature columns remain")
        for column in features:
            try:
                features[column] = pd.to_numeric(features[column], errors="raise")
            except (ValueError, TypeError) as error:
                raise ValueError(f"Non numeric feature: {column}") from error
        if np.isinf(features.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("Features cannot contain infinity")
        if features.isna().all().any():
            raise ValueError("A feature column has no observed values")
    return Dataset(
        path,
        frame,
        features,
        labels,
        target,
        mode,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        excluded,
    )


def validate_pair(train: Dataset, test: Dataset) -> dict:
    if train.fingerprint == test.fingerprint:
        raise ValueError("Training and evaluation files are identical")
    if train.mode != test.mode or set(train.features) != set(test.features):
        raise ValueError("Training and evaluation feature schemas differ")
    unknown = set(test.labels) - set(train.labels)
    if unknown:
        raise ValueError(f"Evaluation has unknown classes: {sorted(unknown)}")
    # Restore training order. Positional models must not silently swap columns.
    test.features = test.features[list(train.features)]
    train_hashes = set(pd.util.hash_pandas_object(train.features, index=False))
    overlap = pd.util.hash_pandas_object(test.features, index=False).isin(train_hashes)
    if train.mode == "text" and overlap.any():
        raise ValueError("Identical text occurs in both training and evaluation splits")
    return {
        "matching_feature_rows": int(overlap.sum()),
        "evaluation_rows": len(test.frame),
        "interpretation": "Feature overlap is reported, not silently removed.",
    }
