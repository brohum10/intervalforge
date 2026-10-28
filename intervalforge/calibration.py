"""Split conformal absolute-residual intervals, pooled or predeclared-grouped."""

from dataclasses import dataclass
import math
from types import MappingProxyType

from .data import name


def threshold(scores, coverage_percent):
    if type(scores) not in (list, tuple):
        raise ValueError("scores must be a list or tuple")
    if type(coverage_percent) is not int or not 1 <= coverage_percent <= 99:
        raise ValueError("coverage_percent must be integer 1-99")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in scores
    ):
        raise ValueError("calibration scores must be finite and nonnegative")
    n = len(scores)
    rank = ((n + 1) * coverage_percent + 99) // 100
    return dict(count=n, rank=rank, radius=sorted(scores)[rank - 1] if rank <= n else None)


@dataclass(frozen=True)
class Calibrated:
    predictor: object
    coverage_percent: int
    pooled: object
    groups: object

    def interval(self, features, group, *, mode="grouped"):
        if not name(group) or type(mode) is not str or mode not in {"pooled", "grouped"}:
            raise ValueError("need a valid group and pooled/grouped mode")
        point = self.predictor.predict(features)
        info = (
            self.pooled
            if mode == "pooled"
            else self.groups.get(group, threshold([], self.coverage_percent))
        )
        radius = info["radius"]
        return dict(
            point=point,
            lower=point - radius if radius is not None else None,
            upper=point + radius if radius is not None else None,
            radius=radius,
            bounded=radius is not None,
            calibration_count=info["count"],
            rank=info["rank"],
            reason=(
                "calibrated"
                if radius is not None
                else (
                    "unseen_group"
                    if mode == "grouped" and group not in self.groups
                    else "insufficient_calibration"
                )
            ),
        )

    def document(self):
        return dict(
            coverage_percent=self.coverage_percent,
            pooled=dict(self.pooled),
            groups={group: dict(info) for group, info in self.groups.items()},
            predictor=self.predictor.document(),
        )


def calibrate(predictor, dataset, *, coverage_percent=90):
    scores, group_scores = [], {}
    if tuple(dataset.document()["feature_names"]) != predictor.feature_names:
        raise ValueError("predictor feature order does not match dataset")
    for row in dataset.partition("calibration"):
        score = abs(row["target"] - predictor.predict(row["features"]))
        scores.append(score)
        group_scores.setdefault(row["group"], []).append(score)
    pooled = MappingProxyType(threshold(scores, coverage_percent))
    groups = MappingProxyType(
        {
            group: MappingProxyType(threshold(values, coverage_percent))
            for group, values in sorted(group_scores.items())
        }
    )
    return Calibrated(predictor, coverage_percent, pooled, groups)
