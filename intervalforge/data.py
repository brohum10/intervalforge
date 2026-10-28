"""Detached, strictly validated train/calibration/test regression records."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

MAX_BYTES = 2_000_000
NAME = re.compile(r"[A-Za-z0-9_.:-]{1,100}\Z")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def number(value):
    return type(value) in (int, float) and abs(value) <= 1e6 and math.isfinite(value)


def name(value):
    return type(value) is str and NAME.fullmatch(value) is not None


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("non-finite JSON constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


@dataclass(frozen=True)
class Dataset:
    encoded: str

    @classmethod
    def from_document(cls, document):
        if type(document) is not dict or set(document) != {"version", "feature_names", "rows"}:
            raise ValueError("dataset needs exactly version/feature_names/rows")
        if type(document["version"]) is not int or document["version"] != 1:
            raise ValueError("version must be integer 1")
        names = document["feature_names"]
        if (
            type(names) is not list
            or not 1 <= len(names) <= 8
            or any(not name(item) for item in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("need 1-8 distinct feature names")
        rows = document["rows"]
        if type(rows) is not list or not 4 <= len(rows) <= 2000:
            raise ValueError("need 4-2000 rows")
        ids, counts, groups, normalized = (
            set(),
            {"train": 0, "calibration": 0, "test": 0},
            set(),
            [],
        )
        for row in rows:
            if type(row) is not dict or set(row) != {"id", "split", "group", "features", "target"}:
                raise ValueError("invalid row fields")
            if not name(row["id"]) or row["id"] in ids or not name(row["group"]):
                raise ValueError("row IDs must be unique; row/group IDs must be valid")
            ids.add(row["id"])
            groups.add(row["group"])
            split = row["split"]
            if type(split) is not str or split not in counts:
                raise ValueError("split must be train, calibration, or test")
            counts[split] += 1
            values = row["features"]
            if (
                type(values) is not list
                or len(values) != len(names)
                or any(not number(x) for x in values)
                or not number(row["target"])
            ):
                raise ValueError(
                    "features/target need finite numbers within +/-1e6; feature count must match names"
                )
            normalized.append(
                dict(
                    id=row["id"],
                    split=split,
                    group=row["group"],
                    features=list(map(float, values)),
                    target=float(row["target"]),
                )
            )
        if (
            counts["train"] < 2
            or not counts["calibration"]
            or not counts["test"]
            or len(groups) > 30
        ):
            raise ValueError("need >=2 train, >=1 calibration, >=1 test row and at most 30 groups")
        encoded = canonical(
            dict(version=1, feature_names=names, rows=sorted(normalized, key=lambda row: row["id"]))
        )
        if len(encoded.encode()) > MAX_BYTES:
            raise ValueError("normalized dataset exceeds byte limit")
        return cls(encoded)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as source:
            data = source.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("dataset exceeds 2 MB")
        return cls.from_document(strict_json(data.decode("utf-8")))

    def document(self):
        return json.loads(self.encoded)

    def partition(self, split):
        if split not in {"train", "calibration", "test"}:
            raise ValueError("unknown partition")
        return [row for row in self.document()["rows"] if row["split"] == split]

    @property
    def sha256(self):
        return hashlib.sha256(self.encoded.encode()).hexdigest()
