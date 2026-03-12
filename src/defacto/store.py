"""Transactional experiment storage and stable reporting exports."""

from __future__ import annotations

import csv
import json
import sqlite3
from importlib.resources import files
from pathlib import Path


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(files("defacto").joinpath("sql/schema.sql").read_text())
    return connection


def save_run(path: str | Path, run: dict) -> None:
    connection = connect(path)
    try:
        with connection:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run["run_id"],
                    run["created_at"],
                    run["model"],
                    run["mode"],
                    run["train"]["rows"],
                    run["test"]["rows"],
                    json.dumps(run, allow_nan=False),
                ),
            )
            connection.executemany(
                "INSERT INTO metrics VALUES (?, ?, ?)",
                [(run["run_id"], key, value) for key, value in run["metrics"].items()],
            )
            connection.executemany(
                "INSERT INTO confusion VALUES (?, ?, ?, ?)",
                [
                    (run["run_id"], actual, predicted, run["confusion"][i][j])
                    for i, actual in enumerate(run["classes"])
                    for j, predicted in enumerate(run["classes"])
                ],
            )
    finally:
        connection.close()


def open_existing(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    if not path.is_file():
        raise ValueError("The experiment database does not exist")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def get_run(path: str | Path, run_id: str | None = None) -> dict:
    connection = open_existing(path)
    try:
        row = connection.execute(
            "SELECT metadata FROM runs WHERE run_id = ?"
            if run_id
            else "SELECT metadata FROM runs ORDER BY created_at DESC, run_id DESC LIMIT 1",
            (run_id,) if run_id else (),
        ).fetchone()
        if row is None:
            raise ValueError("No matching experiment was found")
        return json.loads(row["metadata"])
    finally:
        connection.close()


def safe_cell(value):
    """Keep untrusted labels from becoming formulas when a CSV is opened."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def export_tables(path: str | Path, destination: str | Path) -> list[Path]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    connection = open_existing(path)
    exports = []
    try:
        for name, query in {
            "experiments": "SELECT * FROM experiment_summary ORDER BY created_at, run_id",
            "metrics": "SELECT * FROM metrics ORDER BY run_id, metric",
            "confusion": "SELECT * FROM confusion ORDER BY run_id, actual, predicted",
        }.items():
            cursor = connection.execute(query)
            output = destination / f"{name}.csv"
            with output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow([column[0] for column in cursor.description])
                writer.writerows([[safe_cell(cell) for cell in row] for row in cursor])
            exports.append(output)
    finally:
        connection.close()
    return exports
