# IntervalForge

[![Checks](https://github.com/brohum10/intervalforge/actions/workflows/ci.yml/badge.svg)](https://github.com/brohum10/intervalforge/actions/workflows/ci.yml)

**A point estimate hides the uncertainty. An average coverage number can hide it too.**

IntervalForge is an offline regression-and-calibration lab. It fits a small
ridge regressor, adds split-conformal prediction intervals, and compares one
pooled calibration radius with separate radii for predeclared groups. Reports
show coverage, interval width, subgroup behavior, insufficient-data cases, and
what breaks under a deliberately shifted test distribution.

Python 3.12+ · standard library only · no API key · MIT

## Run it

From a cloned repository folder:

```sh
python -m intervalforge demo
python scripts/verify_reference.py
python -m unittest discover -s tests -v
```

Use `python3` if that is your Python 3.12+ command. The demo needs no account,
downloads, pretrained model, server, or external data. Optional installation:
`python -m pip install .`, then use `intervalforge` instead of
`python -m intervalforge`. Packaging may download build tooling; the runtime
has no third-party dependencies.

## What the synthetic experiment found

The fixture has 200 training, 200 calibration, and 200 test rows. The target is
`2 + 3*x + uniform noise`; the prespecified `quiet` group has noise within ±0.75,
and `noisy` within ±6. Group, feature, and noise are sampled before the target
is formed. This simple linear generator favors the ridge point model; it is
not a real-data forecasting benchmark or a blinded study.

Target coverage is 90%. The same frozen predictor is used for both methods:

| Calibration | Test coverage | Quiet-group coverage | Noisy-group coverage | Mean finite width |
| --- | ---: | ---: | ---: | ---: |
| Pooled | 86.0% | 100.0% | 71.4% | 8.855 |
| Grouped | 91.0% | 93.1% | 88.8% | 6.235 |

The pooled interval is unnecessarily wide for quiet examples but misses many
noisy ones. Grouped calibration uses a wider band for the noisy group and a
narrower band for the quiet group. **It does not improve the point predictor**:
both methods have identical test RMSE, approximately 2.590.

This is one finite synthetic test sample. The 90% target is **not a promise that
90% of every fixed test batch, subgroup sample, or individual prediction will
be covered**. Even the grouped noisy slice here is below target.

### The failure cases stay in the report

- **Noise shift:** freeze the fitted model and calibration, then triple the
  synthetic test noise. Pooled coverage drops to 60.5%; grouped drops to 33.5%.
  No retraining, recalibration, or automatic shift correction is performed.
  These are stress transformations of the same test rows, not a new IID study.
- **Unseen group:** grouped mode emits an unbounded interval for both new-group
  examples. Its nominal coverage on those rows is trivially 100%, but bounded
  coverage and mean finite width are `null`. That is not useful model success.
- **Too little calibration:** at 90%, eight calibration scores require an
  unbounded radius, while nine permit the maximum observed score. The tool does
  not silently clamp the requested quantile to manufacture a finite interval.

[evidence/synthetic.json](evidence/synthetic.json) contains the compact report:
model parameters, calibration counts/ranks/radii, every aggregate and group
metric, shift results, and both unseen-group predictions. `experiment` outputs
**every test row's prediction and interval**, rather than selected successes.
The evidence verifier compares all compact-report fields with floating-point
tolerance `rtol=1e-9, atol=1e-10`; identifiers and hashes must match exactly.

## How it works

```text
train rows       -> train-only scaling + ridge point predictor
calibration rows -> absolute residuals -> pooled/group-specific order statistics
test rows        -> frozen intervals -> coverage/width + subgroup diagnostics
new point        -> point estimate + interval, or explicit unbounded result
```

The ridge model uses standardized training features, an unpenalized intercept,
and the objective `mean squared error + ridge * sum(weights²)`. A small Cholesky
solver handles the positive-definite normal equations. The default penalty is
0.01; it is fixed, not tuned using calibration or test outcomes. Group IDs are
not automatically included as features in the point model.

For absolute calibration residuals `s`, target percentage `p`, and `n` scores:

```text
k = ceil((n + 1) * p / 100)    # evaluated using integer arithmetic
q = kth smallest score       # one-indexed; +infinity if k > n
interval(x) = [prediction(x) - q, prediction(x) + q]
```

Grouped mode applies that rule within the caller's prespecified group. It does
not silently fall back to the pooled radius for an unseen group. Interval
endpoints are inclusive; ties may make coverage conservative.

The established split-conformal coverage result needs calibration and future
scores to be exchangeable with a predictor fixed independently of calibration.
For the group-wise interpretation, groups must be fixed before inspecting
calibration outcomes, available at inference, and support the corresponding
within-group sampling assumptions. These are marginal or group-level
statements over repeated sampling—not arbitrary per-feature/individual
conditional coverage. Shifts and dependent time series can violate them.
See [Angelopoulos and Bates's introduction, including group-balanced calibration](https://arxiv.org/html/2107.07511v6).
This repository is an independent learning implementation of established ideas,
not a new conformal method or a numerical/statistical certification.

## Use a labeled dataset and predict a new point

Export a complete schema example to a **new** filename:

```sh
python -m intervalforge fixture > new-dataset.json
python -m intervalforge experiment new-dataset.json --coverage 90 --ridge 0.01
python -m intervalforge predict new-dataset.json --group noisy --features 1.5
```

The top-level fields are `version: 1`, `feature_names`, and `rows`. Each row has
`id`, `split` (`train`, `calibration`, or `test`), `group`, `features` in declared
order, and numeric `target`. All three partitions are required by this lab's
dataset format. `predict` uses only its training/calibration partitions; the
new point needs no target. It refits on each call; no deployed model service or
serialized-model loader is implemented. Library functions `fit`, `calibrate`,
and `Calibrated.interval` support reuse in one Python process.

The CLI does not write files or contact a network. Shell redirection can
overwrite files, so choose output paths carefully. Successful commands emit
strict JSON and exit 0; invalid input/file errors emit stderr and exit 2.

### Reading unbounded results correctly

`bounded: false`, `lower: null`, and `upper: null` mean **the whole real line**,
not a missing measurement or a narrow interval around the point. `reason`
distinguishes unseen groups from insufficient calibration. Reports always show
bounded/unbounded row counts. Mean finite width excludes unbounded rows and is
`null` if none are bounded; bounded-only coverage follows the same convention.
Do not compare finite widths without also examining the unbounded fraction.

## Tests and evidence boundaries

The 35 tests include Cholesky solution/residual checks across 1–8 dimensions, closed-form ridge
comparison, train-only preprocessing invariance, test-data calibration
invariance, exact integer quantile boundaries, inclusive interval endpoints,
unknown/tiny groups, immutable thresholds, JSON validation, and CLI behavior.
An exact leave-one-out rank enumeration checks all held-out positions for pool
sizes 2–25 at seven coverage levels; it is an algorithm sanity check, not an
empirical proof of a production coverage guarantee. CI runs Linux/Windows with
Python 3.12/3.14 and reproduces the compact evidence.

Limits: 2 MB input, 4–2,000 rows, 1–8 features, 30 groups, numeric magnitudes at
most 1e6, at least two training and one calibration/test row. Target coverage
is an integer 1–99; ridge must be between 1e-6 and 100. Near-constant feature
scales use 1. The complete dataset and predictions stay in memory. ID uniqueness
does not detect aliases, entity leakage, target-derived features, or duplicated
underlying records. The tool does not validate exchangeability for you.

Row/group IDs and feature names use `[A-Za-z0-9_.:-]{1,100}`. Unknown fields,
duplicate JSON keys, bool-valued numbers, and non-finite values are rejected.

No real-data quality, fairness, deployment readiness, or business impact is
claimed. There is no automatic drift detection, time-series conformal method,
cross-validation search, asymmetric interval method, or online recalibration.
See [SECURITY.md](SECURITY.md). Built as a learning/portfolio project with AI assistance.
