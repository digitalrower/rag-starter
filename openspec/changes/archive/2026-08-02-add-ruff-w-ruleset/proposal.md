## Why

The ruff lint config (`[tool.ruff.lint].select` in `pyproject.toml`) currently enables `E, F, I, B, UP` but omits `W` (pycodestyle warnings), so issues like trailing whitespace, blank lines with whitespace, and tab/space mixing pass CI silently. This is a repo-wide policy change: it alters what `ruff check` treats as a CI-blocking failure across every file the lint job covers, so it needs acceptance criteria, not just a config edit.

## What Changes

- Add `"W"` to `[tool.ruff.lint].select` in `pyproject.toml`.
- Run `ruff check --fix` and `ruff format` across `src/`, `evals/`, `scripts/`, `tests/` to clear newly-flagged `W` violations.
- For any `W` violation `--fix` cannot resolve, add a scoped `per-file-ignores` entry with a comment explaining why the rule doesn't apply there, rather than leaving it unfixed or ignoring it repo-wide.
- **BREAKING**: `ruff check` (the CI `lint-and-typecheck` job) will now fail on `W`-class violations that previously passed silently.

## Capabilities

### New Capabilities
- `lint-ruleset`: The set of ruff rule categories enforced as a CI-blocking gate, and the policy for handling violations that can't be auto-fixed.

### Modified Capabilities
(none)

## Impact

- `pyproject.toml` (`[tool.ruff.lint].select`, possibly `[tool.ruff.lint.per-file-ignores]`)
- Any source files under `src/`, `evals/`, `scripts/`, `tests/` that currently trip `W` rules
- CI (`.github/workflows/ci.yml`) — the `lint-and-typecheck` job's `ruff check` step now enforces the expanded ruleset
