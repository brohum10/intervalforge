import argparse
import json
from pathlib import Path
import sys

from . import Dataset, calibrate, experiment, fit
from .synthetic import demo_dataset, reproduce


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train-only ridge and split-conformal prediction-interval experiments."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo")
    commands.add_parser("fixture")
    run = commands.add_parser("experiment")
    run.add_argument("dataset", type=Path)
    predict = commands.add_parser("predict")
    predict.add_argument("dataset", type=Path)
    predict.add_argument("--group", required=True)
    predict.add_argument("--features", type=float, nargs="+", required=True)
    for command in (run, predict):
        command.add_argument(
            "--coverage", type=int, default=90, help="integer target coverage percent, 1-99"
        )
        command.add_argument("--ridge", type=float, default=0.01)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = reproduce()
        elif args.command == "fixture":
            result = demo_dataset().document()
        elif args.command == "experiment":
            result = experiment(
                Dataset.load(args.dataset), coverage_percent=args.coverage, ridge=args.ridge
            )
        else:
            data = Dataset.load(args.dataset)
            calibrated = calibrate(
                fit(data, ridge=args.ridge), data, coverage_percent=args.coverage
            )
            result = {
                mode: calibrated.interval(args.features, args.group, mode=mode)
                for mode in ("pooled", "grouped")
            }
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, OverflowError, RecursionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
