## Purpose

Defines the set of ruff rule categories enforced as a CI-blocking gate, and the policy for handling violations that automated fixing can't resolve.

## ADDED Requirements

### Requirement: W ruleset is a CI-blocking gate
The system SHALL include `"W"` (pycodestyle warnings) in `[tool.ruff.lint].select`, and CI's `lint-and-typecheck` job SHALL fail whenever `ruff check` reports an unresolved `W`-class violation anywhere in `src/`, `evals/`, `scripts/`, or `tests/`.

#### Scenario: Clean codebase passes
- **WHEN** `ruff check src/ evals/ scripts/ tests/` runs against the repo with `W` enabled and no outstanding violations
- **THEN** the command exits zero and CI's lint step succeeds

#### Scenario: New W violation introduced
- **WHEN** a commit introduces a `W`-class violation (e.g. trailing whitespace) in a file not covered by a per-file ignore for that rule
- **THEN** `ruff check` exits non-zero and CI's `lint-and-typecheck` job fails, blocking merge to `main`

### Requirement: Non-auto-fixable W violations require an explicit, documented ignore
A `W` violation that `ruff check --fix` cannot resolve SHALL NOT be left as a silent failure or suppressed by removing `W` from `select`. It SHALL be suppressed only via a scoped `[tool.ruff.lint.per-file-ignores]` entry for that specific file and rule code, accompanied by a comment explaining why the rule doesn't apply.

#### Scenario: Auto-fixable violation is fixed, not ignored
- **WHEN** a `W`-class violation in a file can be resolved by `ruff check --fix`
- **THEN** the violation SHALL be corrected in place; it SHALL NOT be added to `per-file-ignores`

#### Scenario: Non-auto-fixable violation gets a scoped, rationale-documented ignore
- **WHEN** a `W`-class violation remains after running `ruff check --fix` and manual review
- **THEN** it SHALL be suppressed via a `per-file-ignores` entry naming the specific file and rule code, with an adjacent comment in `pyproject.toml` explaining the rationale — not via a blanket repo-wide ignore and not left unaddressed

#### Scenario: Violation left unaddressed fails the gate
- **WHEN** a `W`-class violation exists in a file with no corresponding `per-file-ignores` entry and was not auto-fixed
- **THEN** `ruff check` continues to report it as a failure and CI's lint step continues to fail until it is either fixed or given a documented ignore
