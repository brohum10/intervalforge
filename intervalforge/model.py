"""Small train-only ridge regression using a checked Cholesky solve."""

from dataclasses import dataclass
import hashlib
import math

from .data import canonical, number


def cholesky_solve(matrix, vector):
    n = len(vector)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j] - math.fsum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("ridge system is not numerically positive definite")
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    y = [0.0] * n
    for i in range(n):
        y[i] = (vector[i] - math.fsum(lower[i][k] * y[k] for k in range(i))) / lower[i][i]
    answer = [0.0] * n
    for i in reversed(range(n)):
        answer[i] = (y[i] - math.fsum(lower[k][i] * answer[k] for k in range(i + 1, n))) / lower[i][
            i
        ]
    if any(not math.isfinite(value) for value in answer):
        raise ValueError("non-finite ridge solution")
    return answer


@dataclass(frozen=True)
class Predictor:
    feature_names: tuple
    means: tuple
    scales: tuple
    weights: tuple
    intercept: float
    training_sha256: str

    def predict(self, features):
        if (
            type(features) not in (list, tuple)
            or len(features) != len(self.weights)
            or any(not number(x) for x in features)
        ):
            raise ValueError("invalid prediction feature vector")
        return self.intercept + math.fsum(
            w * (x - mean) / scale
            for w, x, mean, scale in zip(self.weights, features, self.means, self.scales)
        )

    def document(self):
        return dict(
            feature_names=list(self.feature_names),
            means=list(self.means),
            scales=list(self.scales),
            weights=list(self.weights),
            intercept=self.intercept,
            training_sha256=self.training_sha256,
        )


def fit(dataset, *, ridge=0.01):
    if not number(ridge) or not 1e-6 <= ridge <= 100:
        raise ValueError("ridge must be finite and in [1e-6,100]")
    rows = dataset.partition("train")
    names = dataset.document()["feature_names"]
    n, p = len(rows), len(names)
    means = [math.fsum(row["features"][i] for row in rows) / n for i in range(p)]
    deviations = [
        math.sqrt(math.fsum((row["features"][i] - means[i]) ** 2 for row in rows) / n)
        for i in range(p)
    ]
    scales = [value if value >= 1e-12 else 1.0 for value in deviations]
    x = [
        [(value - mean) / scale for value, mean, scale in zip(row["features"], means, scales)]
        for row in rows
    ]
    center = math.fsum(row["target"] for row in rows) / n
    matrix = [
        [math.fsum(row[i] * row[j] for row in x) / n + (ridge if i == j else 0) for j in range(p)]
        for i in range(p)
    ]
    vector = [
        math.fsum(values[i] * (row["target"] - center) for values, row in zip(x, rows)) / n
        for i in range(p)
    ]
    weights = cholesky_solve(matrix, vector)
    fingerprint = hashlib.sha256(
        canonical(dict(feature_names=names, rows=rows, ridge=ridge)).encode()
    ).hexdigest()
    return Predictor(tuple(names), tuple(means), tuple(scales), tuple(weights), center, fingerprint)
