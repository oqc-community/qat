# GitHub Copilot Instructions

## Project overview

This is **QAT** (Quantum Assembly Toolkit), a low-level quantum compiler/runtime package published
to PyPI as `qat-compiler`.

- Language: **Python 3.10–3.12**
- Package manager: **Poetry 2.x**
- Linter/formatter: **Ruff** (line length 92)
- Test framework: **pytest** with `pytest-xdist` for parallel runs
- Versioning: PEP 440, driven by `poetry-dynamic-versioning` from git tags

## Repository layout

- `src/qat/`: compiler/runtime source code.
  - Key areas: `backend/`, `frontend/`, `middleend/`, `pipelines/`, `runtime/`, `ir/`, `model/`,
    `engines/`, `utils/`.
  - Legacy areas: `purr/` and `waveform_v1/` are deprecated/legacy. Note: `waveform_v1` is not a
    top-level package — it lives under `backend/waveform_v1/`, `engines/waveform_v1/`, and
    `pipelines/purr/waveform_v1/`.
- `tests/unit/`: unit tests mirroring `src/qat/` layout; use this as the default location for new
  tests.
- `tests/files/`: static fixtures (hardware, calibrations, QASM/QIR, config inputs).
- `docs/source/`: Sphinx docs and tutorials.
- `benchmarks/`: performance benchmarking code.
- `README.rst`: human-readable instructions and project overview.

Legacy policy note: do not add new code to `src/qat/purr/` unless fixing a regression or doing
refactoring required by dependent core code changes.

## Architecture patterns

- **Pass-based pipeline**: implement passes via `PassConcept` / `PassInfoMixin`, compose with
  `PassManager`, and run via `.run(ir, res_mgr, met_mgr, ...)`.
- **Result/metrics flow**: always thread `ResultManager` and `MetricsManager`; do not keep
  intermediate pipeline state in pass instances.
- **Runtime model**: `BaseRuntime`/`SimpleRuntime` own a `NativeEngine`, optional
  `results_pipeline`, and `ConnectionMode`; execution is batched across `Executable.programs`.
- **IR and deprecation**: IR instructions are Pydantic models (prefer `numpydantic` arrays); treat
  `purr` and WaveformV1 as legacy.

## Code style

- Follow Ruff rules according to `pyproject.toml`.
- Line length: **92 characters**.
- Docstrings: use the repo's reST field-list style, wrapped at 92 characters (`docformatter`).
  Document parameters with `:param name:` and return values with `:returns:`.
- Type hints: required on all public method signatures.
- Imports: `isort` order enforced — stdlib, third-party, `qat.*` (first-party), then `tests.*` and
  `benchmarks.*` (local). No wildcard imports. Note: `benchmarks/` is excluded from Ruff checks
  (`extend-exclude = ["benchmarks/**"]`), so import ordering is not enforced on files inside
  `benchmarks/` itself — only on files elsewhere that import from `benchmarks.*`.
- All new Python files must include the SPDX header. Set the copyright year or year range to match
  the file's creation/update years, e.g.:
  ```python
  # SPDX-License-Identifier: BSD-3-Clause
  # Copyright (c) 2025 Oxford Quantum Circuits Ltd
  ```

### Git commit messages (conventional commit format)

- Format: `<type>(<scope>): <short summary>`
- `type`: `feat|fix|refactor|test|docs|chore|perf|ci`; `summary` imperative, lowercase, no trailing
  period, \<=72 chars.
- One logical change per commit; do not commit directly to `main`.
- If pre-commit modifies files, re-stage and re-commit; do not use `--no-verify`.
- Optional body/footer: explain why, reference issues, include `BREAKING CHANGE:` when needed.

## Quick start

Local developer commands:

- Setup:

  ```bash
  poetry install --with dev
  poetry run pre-commit install
  poetry run format-code
  ```

- Full test suite (including `experimental` and `legacy`, parallel):

  ```bash
  poetry run pytest --experimental-enable --legacy-enable -n 4
  ```

## Dependencies

- Dependency changes require updating `poetry.lock`.
- Do not introduce GPL-licensed dependencies — the CI pip-license check will fail.
- New dependencies must be compatible with Python 3.10–3.12.

## Testing

- Tests live in `tests/unit/` and mirror `src/qat/` layout exactly.
- Use `pytest` fixtures; prefer `pytest-mock` over `unittest.mock`.
- Tests must not use `print` or leave debug statements.
- Parametrise with `@pytest.mark.parametrize` rather than looping in test bodies.
- Available custom markers: `ab_test`, `experimental`, `legacy`, `qblox`.
- Default run (excludes `experimental` and `legacy`): `poetry run pytest -n 4`
- Full suite run (includes `experimental` and `legacy`):
  `poetry run pytest --experimental-enable --legacy-enable -n 4`
- Run with experimental tests included: `poetry run pytest --experimental-enable -n 4`
- Run only experimental tests: `poetry run pytest --experimental-only`
- Run with legacy tests included: `poetry run pytest --legacy-enable -n 4`
- Run only legacy tests: `poetry run pytest --legacy-only`
- Doctests are enabled, so keep docstring examples correct and runnable.
- Do not suppress warnings with broad filters; add specific entries to `filterwarnings` in
  `pyproject.toml` with a linked ticket comment.

## Pull requests

- All PRs target `main`.
- A PR must pass: Ruff lint, Ruff format, jupytext sync, nbstripout, pip-audit, pip-license check,
  notebook execution/validation (`pytest ... notebooks/ipynb --nbmake`), and the full unit test
  suite including `experimental` and `legacy`
  (`poetry run pytest --experimental-enable --legacy-enable -n 4`).
- All CODEOWNERS are requested as reviewers automatically; at least one maintainer approval is
  required to merge.
- New features should be accompanied by unit tests and, where appropriate, a docstring example.
- Do not merge if `legacy`/`purr` tests are newly broken without a linked ticket.

## What to avoid

- Do not commit notebooks with cell output — `nbstripout` enforces this.
- Do not introduce `print` statements in `src/`; use `get_default_logger()` from
  `qat.purr.utils.logger`.
- Pydantic v2 is in use; do not use v1-style validators or `__fields__`.

## Workflow skills

- Use `.github/copilot/skills/pr-description.md` for PR title/body preparation.
- Use `.github/copilot/skills/jira-ticket.md` for Jira ticket creation and updates.
- Use `.github/copilot/skills/pr-review-threads.md` for review-thread triage.
- If a workflow lesson or gotcha is discovered during a session, suggest adding it to the relevant
  skill file or this document via a follow-up commit.
