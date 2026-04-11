"""Permutation diagnostics for trusted locally trained model artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .data import load_dataset


def explain_model(model_path: str | Path, test_path: str | Path, output: str | Path) -> Path:
    artifact = joblib.load(model_path)
    run, pipeline = artifact["run"], artifact["pipeline"]
    test = load_dataset(
        test_path,
        target=run["test"]["target"],
        mode=run["mode"],
        text_column=run["test"]["features"][0],
        exclude=tuple(run["test"]["excluded_columns"]),
    )
    if test.fingerprint != run["test"]["sha256"]:
        raise ValueError("Explanation requires the exact recorded evaluation dataset")
    test.features = test.features[run["train"]["features"]]
    importance = permutation_importance(
        pipeline,
        test.features,
        test.labels,
        scoring="f1_macro",
        n_repeats=10,
        random_state=run["seed"],
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": test.features.columns,
            "macro_f1_drop_mean": importance.importances_mean,
            "macro_f1_drop_std": importance.importances_std,
        }
    ).sort_values("macro_f1_drop_mean", ascending=False)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / "permutation.csv", index=False)
    metadata = {
        "run_id": run["run_id"],
        "evaluation_sha256": test.fingerprint,
        "repeats": 10,
        "seed": run["seed"],
        "baseline_macro_f1": run["metrics"]["macro_f1"],
        "interpretation": [
            "Scores measure predictive reliance, not causal effects.",
            "Correlated features can mask each other's importance.",
            "Negative values indicate that shuffling improved this measured score.",
            "Use these results for inspection, not tuning against the evaluation set.",
        ],
        "all_finite": bool(np.isfinite(importance.importances).all()),
    }
    (destination / "explanation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination
