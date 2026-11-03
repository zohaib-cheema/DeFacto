"""Command line entry points for the reproducible experiment workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import compare_runs
from .data import load_dataset
from .diagnostics import diagnose_splits, write_diagnostics
from .explain import explain_model
from .model import predict_file, train_experiment
from .report import render_report
from .store import export_tables, get_run, save_run


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="DeFacto experiment and reporting tools")
    sub = root.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Check a dataset before training")
    train = sub.add_parser("train", help="Train and evaluate a reproducible model")
    audit = sub.add_parser("audit", help="Compare split quality and feature distributions")
    for command in (inspect, train, audit):
        command.add_argument("--train", required=True)
        command.add_argument("--target", default="Label")
        command.add_argument("--mode", choices=["numeric", "text"], default="numeric")
        command.add_argument("--text-column", default="Statement")
        command.add_argument("--exclude", nargs="*", default=["Search"])
    train.add_argument("--test", required=True)
    train.add_argument("--model", choices=["logistic", "forest", "mlp"], default="logistic")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--output", default="artifacts")
    train.add_argument("--database", default="artifacts/experiments.sqlite")
    audit.add_argument("--test", required=True)
    audit.add_argument("--output", default="artifacts/diagnostics.json")
    report = sub.add_parser("report", help="Regenerate a saved experiment report")
    report.add_argument("--database", default="artifacts/experiments.sqlite")
    report.add_argument("--run-id")
    report.add_argument("--output", default="artifacts/report.html")
    export = sub.add_parser("export", help="Export SQL views as BI ready CSV tables")
    export.add_argument("--database", default="artifacts/experiments.sqlite")
    export.add_argument("--output", default="artifacts/bi")
    predict = sub.add_parser("predict", help="Predict from a trusted local model")
    predict.add_argument("--model", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    explain = sub.add_parser("explain", help="Measure permutation importance on the recorded split")
    explain.add_argument("--model", required=True)
    explain.add_argument("--test", required=True)
    explain.add_argument("--output", default="artifacts/explanation")
    compare = sub.add_parser("compare", help="Compare compatible experiments")
    compare.add_argument("--database", default="artifacts/experiments.sqlite")
    compare.add_argument("--run-ids", nargs="+", required=True)
    compare.add_argument("--output", default="artifacts/comparison.csv")
    return root


def main(argv: list[str] | None = None) -> int:
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command in {"inspect", "train", "audit"}:
            options = dict(
                target=args.target,
                mode=args.mode,
                text_column=args.text_column,
                exclude=tuple(args.exclude),
            )
            train = load_dataset(args.train, **options)
            if args.command == "inspect":
                print(json.dumps(train.profile(), indent=2))
            elif args.command == "audit":
                print(
                    write_diagnostics(
                        diagnose_splits(train, load_dataset(args.test, **options)), args.output
                    )
                )
            else:
                test = load_dataset(args.test, **options)
                run = train_experiment(
                    train,
                    test,
                    args.output,
                    args.model,
                    args.seed,
                    args.folds,
                )
                save_run(args.database, run)
                directory = Path(run["artifact_directory"])
                (directory / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
                render_report(run, directory / "report.html")
                export_tables(args.database, Path(args.output) / "bi")
                print(
                    json.dumps(
                        {
                            "run_id": run["run_id"],
                            "metrics": run["metrics"],
                            "artifacts": str(directory),
                        },
                        indent=2,
                    )
                )
        elif args.command == "report":
            print(render_report(get_run(args.database, args.run_id), args.output))
        elif args.command == "export":
            print("\n".join(map(str, export_tables(args.database, args.output))))
        elif args.command == "predict":
            print(f"Wrote {predict_file(args.model, args.input, args.output)} predictions")
        elif args.command == "explain":
            print(explain_model(args.model, args.test, args.output))
        elif args.command == "compare":
            print(compare_runs(args.database, args.run_ids, args.output))
    except (ValueError, OSError) as error:
        root.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
