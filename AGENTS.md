# AGENTS.md

Instructions for coding agents and reviewers working on **QAT** (Quantum Assembly Toolkit), a
low-level quantum compiler and runtime published to PyPI as `qat-compiler`. Read by GitHub Copilot
(including code review) and by Claude Code through `CLAUDE.md`. Keep it the only copy.

## Rules

- **`purr` is legacy.** Do not add new code under `src/qat/purr/`, any `purr/` subpackage, or
  `waveform_v1` (`backend/waveform_v1/`, `engines/waveform_v1/`, `pipelines/purr/waveform_v1/`).
  Only regression fixes and refactors forced by core changes go there.
- **The `Pyd` prefix marks the modern class.** The unprefixed re-export is still the legacy one:
  `qat.middleend.DefaultMiddleend` and `qat.backend.DefaultBackend` are legacy;
  `qat.middleend.PydDefaultMiddleend` and `qat.backend.PydWaveformBackend` are current.
- **Lint exemptions are deliberate.** `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml`
  exempts legacy `purr/` paths and `notebooks/` on purpose, and `scripts/**` and
  `tools/partial_package_builder.py` as tracked debt (COMPILER-1079). Do not "fix" those lints in
  passing, and do not add an ignore without a ticket.
- **Test markers come from directories.** `tests/conftest.py::pytest_collection_modifyitems` marks
  tests by path: `qblox/` → `qblox`; `legacy/` or `tests/unit/purr/` → `legacy`; `experimental/` →
  `experimental`. Where a test lives decides when it runs. See [Testing](#testing).
- **Passes are stateless.** Keep per-compilation state in the `ResultManager`, not on the pass.
- **Warnings are errors** (`filterwarnings` in `pyproject.toml`). Allow a new one only with a
  specific `default:` entry and a ticket comment, never a broad filter.
- **Doctests run** (`--doctest-modules` over `src`). Keep docstring examples correct.
- **Every Python file has an SPDX header**, except empty `__init__.py` files. A new file carries the
  current year; when you update an older file, extend it to a range (`2024-2026`):
  ```python
  # SPDX-License-Identifier: BSD-3-Clause
  # Copyright (c) 2024-2026 Oxford Quantum Circuits Ltd
  ```
- Do not add `print` to `src/`; use `get_default_logger()` from `qat.purr.utils.logger`.
- Pydantic **v2** only — no v1-style validators or `__fields__`.
- Never hand-edit a version: `poetry-dynamic-versioning` derives it from git tags.
- Do not hand-edit `docs/source/qat.*.rst`; `sphinx-apidoc` generates them in `build-docs`. Some are
  stale, so do not treat them as evidence of an API.
- Do not commit notebook output (`nbstripout`). Edit both halves of a jupytext pair, or run
  `poetry run jupytext-sync`.

## Personal additions

If an `AGENTS.local.md` exists at the repository root, read it too. It holds one person's own
instructions, is git-ignored, and adds to this file; where the two conflict, this file wins. Claude
Code users can use `CLAUDE.local.md` instead, which Claude Code loads automatically.

## Project

- Python **3.10–3.12**, Poetry **2.x**, Ruff (line length **92**), pytest with `pytest-xdist`.
- `src/qat/` is the package; key areas are `backend/`, `frontend/`, `middleend/`, `pipelines/`,
  `runtime/`, `ir/`, `model/`, `engines/` and `utils/`.
- `tests/unit/` mirrors `src/qat/` exactly and is where new tests go; `tests/files/` holds static
  fixtures (calibrations, hardware, QASM, QIR, qatconfig, compiler config, payloads).
- `docs/source/` holds the Sphinx docs; `docs/source/getting_started/` explains the entry points.
- `benchmarks/` holds performance benchmarks.

## Commands

```bash
poetry install --with dev && poetry run pre-commit install   # setup
poetry run format-code                                       # ruff check --fix && ruff format
poetry run pre-commit run --all-files
poetry run pytest -n 4                                       # default selection, see Testing
poetry run pytest --experimental-enable --legacy-enable -n 4 # all unit tests CI runs
poetry run pytest notebooks/ipynb --nbmake
poetry install --with docs && poetry run build-docs          # docs/build/
poetry run pytest benchmarks/run.py --benchmark-only --benchmark-save="<name>"
```

## Architecture

### Passes

The compiler is pass-based (`qat/core/pass_base.py`). Subclass `AnalysisPass`, `TransformPass`,
`ValidationPass` or `LoweringPass`, implement `run(self, ir, res_mgr, met_mgr, *args, **kwargs)`,
and compose with `PassManager` using `|`. `ResultManager` (`core/result_base.py`) carries analysis
results, looked up by type with `res_mgr.lookup_by_type(...)`; `MetricsManager`
(`core/metrics_base.py`) collects metrics. `qat/middleend/default.py` is the worked example; its
comments record real ordering constraints.

### Pipelines

A `Pipeline` (`pipelines/pipeline.py`) is an immutable bundle of model, target data, frontend,
middleend, backend and runtime. `CompilePipeline` and `ExecutePipeline` are its two halves.

```
source (QASM2/3, OpenPulse, QIR)
  → frontend   → QatIR
  → middleend  → validated, lowered QatIR
  → backend    → Executable[Program]
  → runtime    → NativeEngine.execute(), batched over Executable.programs
  → results pipeline (a PassManager) → readouts
```

Prefer `UpdateablePipeline` (`pipelines/updateable.py`) to building `Pipeline` directly: implement
the static `_build_pipeline(config, model, target_data, engine)` and pair it with a `PipelineConfig`
subclass. Given a `BaseModelLoader`, `.update()` refreshes calibration. `pipelines/waveform/full.py`
is a compact example.

`AutoFrontend` (`frontend/auto.py`) detects the source language by trying each frontend. Modern
pipelines use `AutoFrontendWithFlattenedIR.default_for_pydantic(model)`; `default_for_purr` and
`default_for_legacy` keep legacy paths working.

### Entry points

- `QAT(qatconfig).compile / .execute / .run` (`core/qat.py`) resolves pipelines by name
  (`pipeline="default"`) or takes a pipeline object. The built-in defaults are `echo8/16/32`, with
  `echo32` the default.
- qatconfig YAML (`EXTENSIONS`, `HARDWARE`, `ENGINES`, `PIPELINES`, `COMPILE`, `EXECUTE`) names
  classes by import path: fields in `core/config/session.py` (`QatSessionConfig`), entry types in
  `core/config/descriptions.py`, examples in `tests/files/qatconfig/`. Exactly one pipeline is the
  default; a lone pipeline becomes the default automatically. Settings also read `QAT_`-prefixed
  environment variables.
- `QatExtension.load()` (`qat/extensions.py`) is how external packages register themselves; list
  them under `EXTENSIONS`.

### IR, hardware model and target data

- IR instructions (`qat/ir/instructions.py`) are Pydantic models rooted at `Instruction`;
  `InstructionBlock` nests them and arrays use `numpydantic`. Build programs with
  `InstructionBuilder`. `ir/builder_factory.py` maps a hardware model type to its builder with
  `singledispatch`, so other packages can supply their own gate lowerings.
- `PhysicalHardwareModel` (`model/hardware_model.py`) is the calibrated QPU. The legacy
  `QuantumHardwareModel` lives in `purr/compiler/hardware_models.py`; `model/convert_purr.py`
  converts between them.
- `TargetData` (`model/target_data.py`) describes control-hardware limits and is passed with the
  model to most passes. It is frozen; subclass it per target (`backend/qblox/target_data.py`).
- Model loaders (`model/loaders/`) supply models to pipelines. `HardwareLoaders`
  (`core/pipeline.py`) caches them per session.

### Execution

`Executable` (`executables.py`) is a JSON-serialisable set of `Program`s plus the `AcquireData`
needed to read results. `NativeEngine.execute(program)` (`engines/native.py`) returns
`dict[str, np.ndarray]`. `BaseRuntime` / `SimpleRuntime` (`runtime/`) own the engine, a
`ConnectionMode` and a results pipeline (`runtime/results_pipeline.py:get_results_pipeline`).

### Legacy and experimental

Each component has a current and a legacy half: `backend/waveform/` vs `backend/waveform_v1/`,
`engines/waveform/` vs `engines/waveform_v1/`, `middleend/passes/` vs `middleend/passes/purr/`,
`model/loaders/` vs `model/loaders/purr/`, `pipelines/waveform/` vs `pipelines/purr/` and
`pipelines/legacy/`.

`src/qat/experimental/` holds an xDSL/MLIR structured IR (`experimental/dialect/`) and the pulse→Q1
passes. CI requires **98 %** coverage of it.

## Testing

`pytest_configure` in `tests/conftest.py` turns the path markers into a default selection:

| Marker         | Default  | Change it with                                 |
| -------------- | -------- | ---------------------------------------------- |
| `experimental` | excluded | `--experimental-enable`, `--experimental-only` |
| `legacy`       | excluded | `--legacy-enable`, `--legacy-only`             |
| `qblox`        | included | `--qblox-disable`, `--qblox-only`              |

Passing `-m` with a marker's name turns off the default for that marker only.

A/B benchmarks are separate: `benchmarks/conftest.py` marks tests under `a-b_tests/` as `ab_test`
and skips them unless you pass `--ab-enable` or `--ab-only`.

- Use pytest fixtures and `pytest-mock`, not `unittest.mock`.
- Parametrise with `@pytest.mark.parametrize` rather than looping in a test.
- No `print` or leftover debug statements.

## Code style

- Ruff per `pyproject.toml`; line length 92.
- Docstrings: reST field lists (`:param x:`, `:returns:`), wrapped at 92 by `docformatter`.
- Type hints on all public signatures.
- Imports: stdlib, third-party, `qat.*`, then `tests.*` and `benchmarks.*`. No wildcard imports.
  Re-export in `__init__.py` as `X as X`. `benchmarks/` itself is excluded from Ruff, but files
  elsewhere that import `benchmarks.*` still have import order checked.
- Markdown is formatted by `mdformat --wrap 100`, TOML by `taplo`; `actionlint` checks workflows.

## Commits and pull requests

- Conventional commits: `<type>(<scope>): <summary>`, with type one of
  `feat|fix|refactor|test|docs|chore|perf|ci`. Imperative, lowercase, no trailing period, ≤ 72
  characters. One logical change per commit. Use the body to explain why, and add `BREAKING CHANGE:`
  when needed.
- Never commit to `main`. If pre-commit rewrites files, re-stage and commit again; never use
  `--no-verify`.
- PRs target `main`. CODEOWNERS are requested automatically. Merging needs 2 approvals, resolved
  review threads and an up-to-date branch.
- Required checks: consistency and formatting, stable unit tests on 3.10–3.12, and legacy unit
  tests. These must pass too, though not enforced by the ruleset: `pip-audit`, the licence check (no
  GPL), ≥ 60 % coverage, `--experimental-only` at ≥ 98 % coverage, every notebook under `--nbmake`,
  and the docs build.
- Do not merge newly broken `legacy` tests without a linked ticket.
- Label each PR for the changelog (`.github/release.yml`): `breaking-change`, `enhancement`, `bug`,
  `devops`, `documentation`, `experimental` or `ignore-for-release`.
- New features need unit tests and, where useful, a docstring example.
- Dependency changes update `poetry.lock`, support Python 3.10–3.12 and avoid GPL licences.
  `scripts/check-dependency.sh` rejects `git`, `path` or `url` dependencies at release.

## Workflow skills

Multi-step workflows live in `.github/copilot/skills/`:

- `pr-description.md` — a PR title and body safe for `gh --body-file`.
- `jira-ticket.md` — create or update a Jira ticket.
- `pr-review-threads.md` — triage and resolve review threads.

If you find a workflow gotcha, suggest adding it to the relevant skill or to this file.
