"""Descriptive coverage/width reports, never selecting settings from test outcomes."""

import math

from .calibration import calibrate
from .data import name, number
from .model import fit


def evaluate(calibrated, rows, *, mode):
    predictions = []
    seen = set()
    for row in rows:
        if (
            type(row) is not dict
            or not {"id", "group", "features", "target"} <= set(row)
            or not name(row["id"])
            or row["id"] in seen
            or not number(row["target"])
        ):
            raise ValueError(
                "test rows need unique valid IDs, features, group, and finite bounded targets"
            )
        seen.add(row["id"])
        interval = calibrated.interval(row["features"], row["group"], mode=mode)
        covered = not interval["bounded"] or interval["lower"] <= row["target"] <= interval["upper"]
        predictions.append(
            dict(
                id=row["id"], group=row["group"], target=row["target"], covered=covered, **interval
            )
        )
    if not predictions:
        raise ValueError("evaluation needs test rows")

    def summarize(items):
        bounded = [row for row in items if row["bounded"]]
        return dict(
            rows=len(items),
            bounded_rows=len(bounded),
            unbounded_rows=len(items) - len(bounded),
            coverage_including_unbounded=sum(row["covered"] for row in items) / len(items),
            bounded_coverage=(
                sum(row["covered"] for row in bounded) / len(bounded) if bounded else None
            ),
            mean_finite_width=(
                math.fsum(2 * row["radius"] for row in bounded) / len(bounded) if bounded else None
            ),
            rmse=math.sqrt(
                math.fsum((row["target"] - row["point"]) ** 2 for row in items) / len(items)
            ),
        )

    return dict(
        summary=summarize(predictions),
        by_group={
            group: summarize([row for row in predictions if row["group"] == group])
            for group in sorted({row["group"] for row in predictions})
        },
        predictions=predictions,
    )


def experiment(dataset, *, coverage_percent=90, ridge=0.01):
    predictor = fit(dataset, ridge=ridge)
    calibrated = calibrate(predictor, dataset, coverage_percent=coverage_percent)
    test = dataset.partition("test")
    return dict(
        version=1,
        dataset_sha256=dataset.sha256,
        split_counts={
            split: len(dataset.partition(split)) for split in ("train", "calibration", "test")
        },
        ridge=ridge,
        calibration=calibrated.document(),
        test={mode: evaluate(calibrated, test, mode=mode) for mode in ("pooled", "grouped")},
    )
