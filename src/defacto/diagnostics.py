"""Descriptive split diagnostics, never a substitute for provenance review."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Dataset, validate_pair


def distribution_shift(reference: pd.Series, observed: pd.Series, bins: int = 10) -> dict:
    """Compare numeric distributions using bins determined by training data only."""
    clean_reference = reference.dropna().to_numpy(dtype=float)
    clean_observed = observed.dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(clean_reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        value = float(clean_reference[0])
        edges = np.array([-np.inf, value, np.nextafter(value, np.inf), np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    reference_counts, _ = np.histogram(clean_reference, bins=edges)
    observed_counts, _ = np.histogram(clean_observed, bins=edges)
    # Missing observations are a separate bin, so drift cannot hide missingness.
    reference_counts = np.append(reference_counts, reference.isna().sum())
    observed_counts = np.append(observed_counts, observed.isna().sum())
    reference_share = (reference_counts + 0.5) / (len(reference) + 0.5 * len(reference_counts))
    observed_share = (observed_counts + 0.5) / (len(observed) + 0.5 * len(observed_counts))
    psi = float(
        np.sum((observed_share - reference_share) * np.log(observed_share / reference_share))
    )
    return {
        "population_stability_index": psi,
        "reference_mean": float(np.mean(clean_reference)),
        "observed_mean": float(np.mean(clean_observed)) if len(clean_observed) else None,
        "reference_missing_fraction": float(reference.isna().mean()),
        "observed_missing_fraction": float(observed.isna().mean()),
        "reference_bin_counts": reference_counts.astype(int).tolist(),
        "observed_bin_counts": observed_counts.astype(int).tolist(),
    }


def diagnose_splits(train: Dataset, test: Dataset) -> dict:
    overlap = validate_pair(train, test)
    labels = sorted(set(train.labels) | set(test.labels))
    train_share = train.labels.value_counts(normalize=True)
    test_share = test.labels.value_counts(normalize=True)
    classes = [
        {
            "label": label,
            "training_fraction": float(train_share.get(label, 0)),
            "evaluation_fraction": float(test_share.get(label, 0)),
        }
        for label in labels
    ]
    if train.mode == "numeric":
        features = {
            column: distribution_shift(train.features[column], test.features[column])
            for column in train.features
        }
    else:
        features = {
            "text_length": distribution_shift(
                train.features.iloc[:, 0].str.len(),
                test.features.iloc[:, 0].str.len(),
            ),
            "word_count": distribution_shift(
                train.features.iloc[:, 0].str.split().str.len(),
                test.features.iloc[:, 0].str.split().str.len(),
            ),
        }
    return {
        "train": train.profile(),
        "test": test.profile(),
        "overlap": overlap,
        "class_distribution": classes,
        "feature_distribution": features,
        "method": "Training quantile bins with a separate missing bin and 0.5 count smoothing.",
        "interpretation": [
            "These are descriptive comparisons with no automatic pass threshold.",
            "A distribution difference does not by itself establish a data defect.",
            "Prepared tables cannot establish whether upstream features used future information.",
            "Evaluation features and labels must remain separate from model selection.",
        ],
    }


def write_diagnostics(diagnostics: dict, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8")
    return destination
