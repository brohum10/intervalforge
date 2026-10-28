# Trust and uncertainty boundaries

IntervalForge reads bounded local JSON and prints results. It does not unpickle
models, run feature values as code, train on remote services, write artifacts,
or make network requests. Packaging may download build tooling. Shell output
redirection can overwrite an existing file; choose new paths when exporting.

Dataset IDs, feature values, targets, and prediction records may be sensitive.
Use synthetic or sanitized data in public repositories. Input/training hashes
identify normalized content but are not signatures or proof of provenance.

Feature/row limits bound this small implementation but are not hard CPU or
memory isolation. The in-process ridge solver is an educational numerical
implementation, not a replacement for a production numerical library. Ill-posed
or numerically non-positive-definite systems fail rather than yielding a model.

A requested coverage target is not a promise for an individual outcome or fixed
test batch. Coverage interpretations require defensible sampling assumptions,
an independently fitted predictor, and prespecified groups. The program cannot
detect semantic leakage, invalid group construction, time dependencies, or a
changing population from schema validation alone.

Unbounded intervals are serialized with null endpoints and `bounded: false`.
They must not be interpreted as finite-width success or as zero uncertainty.
The tool does not approve production releases or supply medical, financial,
fairness, or safety-critical guarantees. Verify application assumptions and
use independent validation before any real consequential use.

Use GitHub private vulnerability reporting where available; never include
private targets, identifiers, or secrets in public issues.
