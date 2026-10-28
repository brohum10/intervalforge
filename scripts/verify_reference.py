"""Recompute all compact synthetic evidence; no real-data quality claim."""

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intervalforge.synthetic import reproduce


def compare(expected, actual, path="report"):
    if type(expected) is float:
        if type(actual) not in (float, int) or not math.isclose(
            expected, actual, rel_tol=1e-9, abs_tol=1e-10
        ):
            raise AssertionError(f"numeric mismatch at {path}")
    elif type(expected) is dict:
        if type(actual) is not dict or set(expected) != set(actual):
            raise AssertionError(f"field mismatch at {path}")
        for key in expected:
            compare(expected[key], actual[key], f"{path}.{key}")
    elif type(expected) is list:
        if type(actual) is not list or len(expected) != len(actual):
            raise AssertionError(f"list mismatch at {path}")
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(a, b, f"{path}[{i}]")
    elif type(expected) is not type(actual) or expected != actual:
        raise AssertionError(f"value mismatch at {path}")


def main():
    actual = reproduce()
    expected = json.loads((ROOT / "evidence/synthetic.json").read_text(encoding="utf-8"))
    compare(expected, actual)
    for mode, result in actual["test"].items():
        print(
            f'{mode}: test coverage={result["summary"]["bounded_coverage"]:.3f}; finite width={result["summary"]["mean_finite_width"]:.3f}'
        )
    print(
        "Verified calibration ranks, predictors, group diagnostics, noise-shift stress, and unbounded unseen-group intervals."
    )


if __name__ == "__main__":
    main()
