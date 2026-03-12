"""Compare recorded experiments only when their evaluation context matches."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from .store import get_run


def compare_runs(database: str | Path, run_ids: list[str], destination: str | Path) -> Path:
    if len(set(run_ids)) < 2:
        raise ValueError("Select at least two distinct runs")
    runs = [get_run(database, run_id) for run_id in run_ids]
    contexts = {
        (
            run["train"]["sha256"],
            run["test"]["sha256"],
            run["mode"],
            run["train"]["target"],
            tuple(run["train"]["features"]),
        )
        for run in runs
    }
    if len(contexts) != 1:
        raise ValueError(
            "Runs use different data or feature schemas and cannot be compared directly"
        )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id",
        "model",
        "seed",
        "cv_macro_f1_mean",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "run_id": run["run_id"],
                    "model": run["model"],
                    "seed": run["seed"],
                    **{key: run["metrics"][key] for key in columns[3:]},
                }
            )
    return destination


def class_errors(database: str | Path, run_id: str) -> list[dict]:
    from importlib.resources import files

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        query = files("defacto").joinpath("sql/class_errors.sql").read_text()
        return [dict(row) for row in connection.execute(query, (run_id,))]
    finally:
        connection.close()
