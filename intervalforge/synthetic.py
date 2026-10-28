"""Synthetic heteroskedastic regression, plus explicit assumption-breaking stress cases."""

import random

from .calibration import calibrate
from .data import Dataset
from .experiment import evaluate, experiment
from .model import fit


def demo_dataset():
    rng = random.Random(20261028)
    rows = []
    for split in ("train", "calibration", "test"):
        for i in range(200):
            group = rng.choice(["quiet", "noisy"])
            x = 4 * rng.random() - 2
            scale = 0.75 if group == "quiet" else 6.0
            noise = scale * (2 * rng.random() - 1)
            rows.append(
                dict(
                    id=f"{split}-{i:03}",
                    split=split,
                    group=group,
                    features=[x],
                    target=2 + 3 * x + noise,
                )
            )
    return Dataset.from_document(dict(version=1, feature_names=["workload"], rows=rows))


def compact(result):
    return {key: value for key, value in result.items() if key != "predictions"}


def reproduce():
    data = demo_dataset()
    result = experiment(data)
    calibrated = calibrate(fit(data), data)
    shifted = data.partition("test")
    for row in shifted:
        truth = 2 + 3 * row["features"][0]
        row["target"] = truth + 3 * (row["target"] - truth)
    unknown = [
        dict(id=f"unseen-{i}", split="test", group="unseen", features=[float(i)], target=2 + 3 * i)
        for i in range(2)
    ]
    result["test"] = {mode: compact(value) for mode, value in result["test"].items()}
    result["noise_shift_stress"] = {
        mode: compact(evaluate(calibrated, shifted, mode=mode)) for mode in ("pooled", "grouped")
    }
    result["unseen_group_stress"] = {
        mode: evaluate(calibrated, unknown, mode=mode) for mode in ("pooled", "grouped")
    }
    result["corpus"] = (
        "seed=20261028; y=2+3*x+uniform noise; quiet +/-0.75, noisy +/-6; group chosen before target"
    )
    result["stress_definition"] = (
        "Frozen predictor/calibration; test residual noise multiplied by 3, plus two new-group rows. Not a new IID test study."
    )
    return result
