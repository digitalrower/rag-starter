## 1. Enable the ruleset

- [x] 1.1 Add `"W"` to `[tool.ruff.lint].select` in `pyproject.toml`

## 2. Clear violations

- [x] 2.1 Run `ruff check --fix src/ evals/ scripts/ tests/` and review the diff
- [x] 2.2 Run `ruff format src/ evals/ scripts/ tests/`
- [x] 2.3 Run `ruff check src/ evals/ scripts/ tests/` again and list any remaining `W`-class violations `--fix` couldn't resolve
- [x] 2.4 For each remaining violation, either hand-fix it, or add a scoped `[tool.ruff.lint.per-file-ignores]` entry naming the file and rule code with an adjacent comment explaining why the rule doesn't apply

## 3. Verify

- [x] 3.1 Run `ruff check src/ evals/ scripts/ tests/` and confirm a clean, zero-exit pass
- [x] 3.2 Run `ruff format --check src/ evals/ scripts/ tests/` and confirm no formatting diffs
- [x] 3.3 Run `mypy src/ evals/ scripts/` and `pytest -v` to confirm the fixes didn't change behavior
