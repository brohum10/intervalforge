import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from intervalforge import Dataset, calibrate, evaluate, experiment, fit, threshold
from intervalforge.data import MAX_BYTES, strict_json
from intervalforge.model import cholesky_solve
from intervalforge.synthetic import demo_dataset, reproduce

ROOT = Path(__file__).resolve().parents[1]


def small():
    rows = [
        dict(id="train-a", split="train", group="g", features=[-1, 1], target=-1),
        dict(id="train-b", split="train", group="g", features=[1, 1], target=5),
    ]
    rows += [
        dict(id=f"cal-{i:02}", split="calibration", group="g", features=[0, 1], target=2 + i / 10)
        for i in range(10)
    ]
    rows.append(dict(id="test", split="test", group="g", features=[0, 1], target=2.5))
    return dict(version=1, feature_names=["x", "constant"], rows=rows)


class DataTests(unittest.TestCase):
    def test_detaches_mutable_input_and_output(self):
        doc = small()
        subject = Dataset.from_document(doc)
        doc["rows"].clear()
        subject.document()["rows"].clear()
        self.assertEqual(len(subject.document()["rows"]), 13)

    def test_order_and_integer_float_normalization(self):
        doc = small()
        before = Dataset.from_document(doc)
        doc["rows"].reverse()
        for row in doc["rows"]:
            row["features"] = list(map(float, row["features"]))
            row["target"] = float(row["target"])
        self.assertEqual(before.sha256, Dataset.from_document(doc).sha256)

    def test_schema_and_version(self):
        for value in (True, 1.0, 2, "1"):
            doc = small()
            doc["version"] = value
            with self.assertRaises(ValueError):
                Dataset.from_document(doc)
        doc = small()
        doc["extra"] = 1
        with self.assertRaises(ValueError):
            Dataset.from_document(doc)

    def test_numeric_values_and_shapes(self):
        for values in ([1], [True, 1], [float("nan"), 1], [float("inf"), 1], [1e6 + 1, 1]):
            doc = small()
            doc["rows"][0]["features"] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                Dataset.from_document(doc)
        for target in (True, None, float("inf"), 1e6 + 1):
            doc = small()
            doc["rows"][0]["target"] = target
            with self.assertRaises(ValueError):
                Dataset.from_document(doc)

    def test_unique_ids_and_feature_names(self):
        doc = small()
        doc["rows"][1]["id"] = doc["rows"][0]["id"]
        with self.assertRaises(ValueError):
            Dataset.from_document(doc)
        doc = small()
        doc["feature_names"] = ["x", "x"]
        with self.assertRaises(ValueError):
            Dataset.from_document(doc)

    def test_partition_and_group_requirements(self):
        for changed in ("train", "calibration", "test"):
            doc = small()
            for row in doc["rows"]:
                if row["split"] == changed:
                    row["split"] = "test" if changed != "test" else "train"
            with self.assertRaises(ValueError):
                Dataset.from_document(doc)
        doc = small()
        doc["rows"][0]["group"] = "bad group"
        with self.assertRaises(ValueError):
            Dataset.from_document(doc)

    def test_json_keys_numbers_bytes_and_encoding(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                strict_json(text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            for data in (b" " * (MAX_BYTES + 1), b"\xff"):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    Dataset.load(path)


class RidgeTests(unittest.TestCase):
    def test_seeded_spd_systems_up_to_eight_dimensions(self):
        rng = random.Random(20261028)
        for size in range(1, 9):
            for case in range(3):
                basis = [[2 * rng.random() - 1 for _ in range(size)] for _ in range(size)]
                matrix = [
                    [
                        math.fsum(row[i] * row[j] for row in basis) + (0.2 if i == j else 0)
                        for j in range(size)
                    ]
                    for i in range(size)
                ]
                expected = [2 * rng.random() - 1 for _ in range(size)]
                vector = [math.fsum(a * b for a, b in zip(row, expected)) for row in matrix]
                actual = cholesky_solve(matrix, vector)
                with self.subTest(size=size, case=case):
                    for a, b in zip(actual, expected):
                        self.assertAlmostEqual(a, b, places=11)

    def test_cholesky_known_solution_and_residual(self):
        matrix, vector = [[4.0, 1.0], [1.0, 3.0]], [1.0, 2.0]
        solution = cholesky_solve(matrix, vector)
        self.assertAlmostEqual(solution[0], 1 / 11)
        self.assertAlmostEqual(solution[1], 7 / 11)
        for row, target in zip(matrix, vector):
            self.assertAlmostEqual(
                math.fsum(a * b for a, b in zip(row, solution)), target, places=12
            )

    def test_non_positive_definite_system_rejected(self):
        with self.assertRaises(ValueError):
            cholesky_solve([[0.0]], [1.0])

    def test_closed_form_ridge_and_unpenalized_intercept(self):
        model = fit(Dataset.from_document(small()), ridge=0.01)
        self.assertEqual(model.intercept, 2)
        self.assertAlmostEqual(model.weights[0], 3 / 1.01)
        self.assertEqual(model.weights[1], 0)
        self.assertEqual(model.scales[1], 1)
        self.assertAlmostEqual(model.predict([1, 1]), 2 + 3 / 1.01)

    def test_holdout_changes_cannot_affect_training(self):
        doc = small()
        before = fit(Dataset.from_document(doc)).document()
        for row in doc["rows"]:
            if row["split"] != "train":
                row["features"] = [999, -999]
                row["target"] = -999
        self.assertEqual(before, fit(Dataset.from_document(doc)).document())

    def test_invalid_ridge_and_inference_vectors(self):
        data = Dataset.from_document(small())
        for ridge in (0, -1, True, float("nan"), 101):
            with self.assertRaises(ValueError):
                fit(data, ridge=ridge)
        model = fit(data)
        for values in ([0], [True, 1], [float("inf"), 1]):
            with self.assertRaises(ValueError):
                model.predict(values)

    def test_training_order_is_deterministic(self):
        doc = small()
        before = fit(Dataset.from_document(doc)).document()
        doc["rows"].reverse()
        self.assertEqual(before, fit(Dataset.from_document(doc)).document())


class QuantileTests(unittest.TestCase):
    def test_n_plus_one_rank_not_naive_percentile(self):
        result = threshold(list(range(1, 11)), 90)
        self.assertEqual(result, dict(count=10, rank=10, radius=10))

    def test_nine_points_suffice_but_eight_are_unbounded_at_ninety(self):
        self.assertEqual(threshold(list(range(9)), 90)["radius"], 8)
        self.assertIsNone(threshold(list(range(8)), 90)["radius"])

    def test_empty_and_singleton(self):
        self.assertEqual(threshold([], 90), dict(count=0, rank=1, radius=None))
        self.assertIsNone(threshold([0], 90)["radius"])
        self.assertEqual(threshold([7], 50)["radius"], 7)

    def test_ties_and_zero_residuals(self):
        self.assertEqual(threshold([0] * 20, 90)["radius"], 0)
        self.assertEqual(threshold([1] * 20, 90)["radius"], 1)

    def test_exact_integer_boundary_avoids_float_ceil_errors(self):
        self.assertEqual(threshold(list(range(99)), 99)["rank"], 99)
        self.assertEqual(threshold(list(range(19)), 95)["rank"], 19)
        self.assertEqual(threshold(list(range(4)), 80)["rank"], 4)

    def test_input_not_mutated(self):
        scores = [3, 1, 2]
        threshold(scores, 50)
        self.assertEqual(scores, [3, 1, 2])

    def test_invalid_scores_and_coverage(self):
        for scores in ([True], [-1], [float("inf")], [float("nan")], "scores"):
            with self.assertRaises(ValueError):
                threshold(scores, 90)
        for percent in (0, 100, True, 90.0, "90"):
            with self.assertRaises(ValueError):
                threshold([1], percent)

    def test_leave_one_out_rank_coverage_enumeration(self):
        # Exact finite-pool check over every held-out position; no Monte Carlo tolerance.
        for size in range(2, 26):
            for percent in (1, 10, 50, 80, 90, 95, 99):
                covered = 0
                values = list(range(size))
                for index, test in enumerate(values):
                    radius = threshold(values[:index] + values[index + 1 :], percent)["radius"]
                    covered += radius is None or test <= radius
                with self.subTest(size=size, percent=percent):
                    self.assertEqual(covered, ((size * percent) + 99) // 100)


class CalibrationTests(unittest.TestCase):
    def calibrated(self):
        data = Dataset.from_document(small())
        return calibrate(fit(data), data)

    def test_known_group_threshold(self):
        model = self.calibrated()
        interval = model.interval([0, 1], "g")
        self.assertAlmostEqual(interval["radius"], 0.9)
        self.assertEqual(interval["calibration_count"], 10)
        self.assertTrue(interval["bounded"])

    def test_unknown_group_is_not_silently_pooled(self):
        model = self.calibrated()
        grouped = model.interval([0, 1], "new")
        pooled = model.interval([0, 1], "new", mode="pooled")
        self.assertFalse(grouped["bounded"])
        self.assertEqual(grouped["reason"], "unseen_group")
        self.assertIsNone(grouped["lower"])
        self.assertTrue(pooled["bounded"])

    def test_insufficient_known_group_is_unbounded(self):
        doc = small()
        for row in doc["rows"]:
            if row["id"] == "cal-00":
                row["group"] = "tiny"
        data = Dataset.from_document(doc)
        model = calibrate(fit(data), data)
        interval = model.interval([0, 1], "tiny")
        self.assertFalse(interval["bounded"])
        self.assertEqual(interval["reason"], "insufficient_calibration")

    def test_test_changes_do_not_change_predictor_or_calibration(self):
        doc = small()
        data = Dataset.from_document(doc)
        before = calibrate(fit(data), data).document()
        for row in doc["rows"]:
            if row["split"] == "test":
                row["features"] = [1e4, -1e4]
                row["target"] = 1e5
                row["group"] = "novel"
        changed = Dataset.from_document(doc)
        self.assertEqual(before, calibrate(fit(changed), changed).document())

    def test_thresholds_and_export_are_detached(self):
        model = self.calibrated()
        with self.assertRaises(TypeError):
            model.groups["g"]["radius"] = 0
        model.document()["groups"]["g"]["radius"] = 0
        self.assertAlmostEqual(model.interval([0, 1], "g")["radius"], 0.9)

    def test_group_radius_can_differ_without_changing_point_model(self):
        data = demo_dataset()
        model = calibrate(fit(data), data)
        self.assertGreater(model.groups["noisy"]["radius"], model.groups["quiet"]["radius"])
        self.assertEqual(
            model.interval([0], "noisy")["point"], model.interval([0], "quiet")["point"]
        )

    def test_feature_order_and_mode_validation(self):
        doc = small()
        data = Dataset.from_document(doc)
        predictor = fit(data)
        doc["feature_names"].reverse()
        with self.assertRaises(ValueError):
            calibrate(predictor, Dataset.from_document(doc))
        for mode in ("guess", []):
            with self.assertRaises(ValueError):
                self.calibrated().interval([0, 1], "g", mode=mode)


class EvaluationTests(unittest.TestCase):
    def test_boundary_inclusion(self):
        data = Dataset.from_document(small())
        calibrated = calibrate(fit(data), data)
        radius = calibrated.groups["g"]["radius"]
        rows = [
            dict(id="lower", group="g", features=[0, 1], target=2 - radius),
            dict(id="upper", group="g", features=[0, 1], target=2 + radius),
        ]
        result = evaluate(calibrated, rows, mode="grouped")
        self.assertEqual(result["summary"]["bounded_coverage"], 1)

    def test_unbounded_coverage_never_reports_zero_width(self):
        data = Dataset.from_document(small())
        calibrated = calibrate(fit(data), data)
        rows = [dict(id="u", group="unknown", features=[0, 1], target=999)]
        result = evaluate(calibrated, rows, mode="grouped")["summary"]
        self.assertEqual(result["coverage_including_unbounded"], 1)
        self.assertEqual(result["unbounded_rows"], 1)
        self.assertIsNone(result["mean_finite_width"])
        self.assertIsNone(result["bounded_coverage"])

    def test_demo_preserves_noise_shift_failures(self):
        report = reproduce()
        clean = report["test"]["grouped"]["summary"]["bounded_coverage"]
        shifted = report["noise_shift_stress"]["grouped"]["summary"]["bounded_coverage"]
        self.assertGreater(clean, shifted)
        self.assertEqual(report["unseen_group_stress"]["grouped"]["summary"]["unbounded_rows"], 2)

    def test_group_counts_reconcile_and_full_output_contains_all_rows(self):
        report = experiment(demo_dataset())
        for result in report["test"].values():
            self.assertEqual(len(result["predictions"]), 200)
            self.assertEqual(sum(group["rows"] for group in result["by_group"].values()), 200)
        json.dumps(report, allow_nan=False)


class CliTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "intervalforge", *map(str, args)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=15,
        )

    def test_experiment_and_unlabeled_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(small()), encoding="utf-8")
            result = self.cli("experiment", path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("test", json.loads(result.stdout))
            result = self.cli("predict", path, "--group", "new", "--features", 0, 1)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)["grouped"]["bounded"])

    def test_bad_file_and_settings_have_no_partial_json(self):
        result = self.cli("experiment", ROOT / "missing.json")
        self.assertEqual(result.returncode, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(small()), encoding="utf-8")
            for setting in (("--coverage", 100), ("--ridge", 0)):
                result = self.cli("experiment", path, *setting)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
