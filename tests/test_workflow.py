from __future__ import annotations

import csv
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
import pytest

from defacto.cli import main
from defacto.compare import class_errors, compare_runs
from defacto.data import load_dataset, validate_pair
from defacto.diagnostics import diagnose_splits, distribution_shift
from defacto.model import predict_file, train_experiment
from defacto.report import render_report
from defacto.store import export_tables, get_run, save_run


@pytest.fixture
def datasets(tmp_path):
    rng = np.random.default_rng(17)
    paths = []
    for name, rows in [("train", 80), ("test", 24)]:
        x = rng.normal(size=rows)
        path = tmp_path / f"{name}.csv"
        pd.DataFrame(
            {
                "value": x,
                "other": rng.normal(size=rows),
                "Search": np.arange(rows),
                "Label": (x > 0).astype(int),
            }
        ).to_csv(path, index=False)
        paths.append(path)
    return [load_dataset(path) for path in paths]


def test_end_to_end_records_measured_results_and_predicts(datasets, tmp_path):
    train, test = datasets
    run = train_experiment(train, test, tmp_path / "models", folds=3)
    database = tmp_path / "results.sqlite"
    save_run(database, run)
    stored = get_run(database)
    assert stored["run_id"] == run["run_id"]
    assert 0 <= stored["metrics"]["accuracy"] <= 1
    assert sum(map(sum, stored["confusion"])) == len(test.frame)
    assert sum(row["evaluated_rows"] for row in class_errors(database, run["run_id"])) == len(
        test.frame
    )
    outputs = export_tables(database, tmp_path / "bi")
    with outputs[0].open() as stream:
        exported = list(csv.DictReader(stream))
    assert float(exported[0]["accuracy"]) == run["metrics"]["accuracy"]
    model = tmp_path / "models" / run["run_id"] / "model.joblib"
    assert predict_file(model, test.path, tmp_path / "predictions.csv") == len(test.frame)
    report = render_report(stored, tmp_path / "report.html")
    assert "Classification experiment" in report.read_text()


def test_preprocessing_is_fit_on_training_only(datasets, tmp_path):
    train, test = datasets
    test.features["value"] = 100000
    run = train_experiment(train, test, tmp_path, folds=2)
    artifact = joblib.load(tmp_path / run["run_id"] / "model.joblib")
    scaler = artifact["pipeline"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(train.features["value"].mean())
    assert "Search" not in run["train"]["features"]


def test_reordered_columns_produce_same_schema(datasets):
    train, test = datasets
    test.features = test.features[list(reversed(test.features.columns))]
    validate_pair(train, test)
    assert list(test.features) == list(train.features)


def test_unknown_labels_and_schema_mismatches_are_rejected(datasets):
    train, test = datasets
    test.labels.iloc[0] = "unexpected"
    with pytest.raises(ValueError, match="unknown classes"):
        validate_pair(train, test)
    test.features["leak"] = test.labels
    with pytest.raises(ValueError, match="schemas differ"):
        validate_pair(train, test)


def test_same_file_cannot_be_used_as_evaluation(datasets):
    with pytest.raises(ValueError, match="identical"):
        validate_pair(datasets[0], datasets[0])


@pytest.mark.parametrize(
    "data,message",
    [
        ({"value": [1, 2], "Label": [None, "a"]}, "nonempty target"),
        ({"value": [1, 2], "Label": ["a", "a"]}, "two target classes"),
        ({"value": ["x", "y"], "Label": ["a", "b"]}, "Non numeric"),
        ({"value": [None, None], "Label": ["a", "b"]}, "no observed"),
        ({"value": [float("inf"), 2], "Label": ["a", "b"]}, "infinity"),
    ],
)
def test_invalid_training_inputs_fail_early(data, message, tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_dataset(path)


def test_text_mode_runs_without_claiming_prepared_tables_are_articles(tmp_path):
    paths = []
    for name, suffix in [("train", "training"), ("test", "evaluation")]:
        rows = [
            {"Statement": f"{word} topic example {suffix} {i}", "Label": label}
            for label, word in [("a", "apple"), ("b", "orange")]
            for i in range(12)
        ]
        path = tmp_path / f"{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        paths.append(path)
    train, test = [load_dataset(path, mode="text") for path in paths]
    run = train_experiment(train, test, tmp_path / "models", folds=3)
    assert run["mode"] == "text"
    test.features.iloc[0, 0] = train.features.iloc[0, 0]
    with pytest.raises(ValueError, match="Identical text"):
        validate_pair(train, test)


def test_report_escapes_untrusted_labels_and_sql_is_parameterized(datasets, tmp_path):
    run = train_experiment(*datasets, tmp_path / "models", folds=2)
    run["classes"][0] = "<script>alert(1)</script>"
    text = render_report(run, tmp_path / "report.html").read_text()
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    save_run(tmp_path / "runs.sqlite", run)
    with pytest.raises(ValueError, match="No matching"):
        get_run(tmp_path / "runs.sqlite", "' OR 1=1 --")


def test_duplicate_run_save_rolls_back_without_partial_rows(datasets, tmp_path):
    run = train_experiment(*datasets, tmp_path / "models", folds=2)
    database = tmp_path / "runs.sqlite"
    save_run(database, run)
    with pytest.raises(sqlite3.IntegrityError):
        save_run(database, run)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == len(
            run["metrics"]
        )


def test_feature_shift_includes_missingness_and_is_finite(datasets):
    train, test = datasets
    result = diagnose_splits(train, test)
    assert len(result["feature_distribution"]) == 2
    shift = distribution_shift(pd.Series([1, 1, 1]), pd.Series([1, None, 2]))
    assert np.isfinite(shift["population_stability_index"])
    assert shift["observed_missing_fraction"] == pytest.approx(1 / 3)


def test_comparison_rejects_different_data(datasets, tmp_path):
    database = tmp_path / "runs.sqlite"
    first = train_experiment(*datasets, tmp_path / "models", folds=2)
    second = train_experiment(*datasets, tmp_path / "models", folds=2, seed=7)
    save_run(database, first)
    save_run(database, second)
    ids = [first["run_id"], second["run_id"]]
    assert compare_runs(database, ids, tmp_path / "compare.csv").exists()
    second["test"]["sha256"] = "changed"
    third = dict(second, run_id="different")
    save_run(database, third)
    with pytest.raises(ValueError, match="different data"):
        compare_runs(database, [first["run_id"], third["run_id"]], tmp_path / "bad.csv")


def test_cli_generates_artifacts(datasets, tmp_path, capsys):
    database = tmp_path / "results.sqlite"
    assert (
        main(
            [
                "train",
                "--train",
                str(datasets[0].path),
                "--test",
                str(datasets[1].path),
                "--folds",
                "2",
                "--output",
                str(tmp_path / "artifacts"),
                "--database",
                str(database),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert (tmp_path / "artifacts" / output["run_id"] / "run.json").exists()


def test_csv_exports_neutralize_formula_labels(datasets, tmp_path):
    run = train_experiment(*datasets, tmp_path / "models", folds=2)
    run["classes"][0] = "=HYPERLINK(1)"
    database = tmp_path / "results.sqlite"
    save_run(database, run)
    exports = export_tables(database, tmp_path / "bi")
    assert "'=HYPERLINK(1)" in exports[2].read_text()


def test_duplicate_headers_are_rejected_before_pandas_can_rename_them(tmp_path):
    path = tmp_path / "ambiguous.csv"
    path.write_text("Label,Label,value\na,a,1\nb,b,2\n")
    with pytest.raises(ValueError, match="unique"):
        load_dataset(path)


def test_read_commands_do_not_create_missing_databases(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(ValueError, match="does not exist"):
        get_run(path)
    assert not path.exists()
