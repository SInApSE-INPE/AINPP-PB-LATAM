# STORY-003: Standardized Pytest & Quality Gates

**Status:** Draft
**Assignee:** @qa
**Epic:** Framework Evolution

## Story
**As a** Developer,
**I want** standard Pytest patterns and quality gates (linting, type checking) for the new package structure,
**so that** I can maintain high code quality and prevent regressions in scientific logic.

## Acceptance Criteria
- [ ] All existing tests are refactored to use the new `ainpp` namespace.
- [ ] Pytest is configured via `pyproject.toml` (or `pytest.ini`).
- [ ] `pytest-cov` is integrated for code coverage reporting.
- [ ] `flake8` and `mypy` configurations are added to `pyproject.toml`.
- [ ] The `tests/` directory is updated with proper fixtures for datasets and models.
- [ ] `run_tests.sh` is deprecated in favor of standardized CLI commands.

## 🤖 CodeRabbit Integration
**Story Type Analysis**: QA / Architecture
**Complexity**: Medium
**Specialized Agent Assignment**:
- Primary Agents: @qa, @dev
- Supporting Agents: @github-devops

**Quality Gate Tasks**:
- [ ] Pre-PR (@github-devops): Verify test coverage and static analysis.

**CodeRabbit Focus Areas**:
- Test modularity and fixture reuse.
- Correctness of coverage reports for nested modules.
- Proper exclusion patterns for static analysis (`mypy`, `flake8`).

## Tasks / Subtasks
- [ ] Update imports in all files in `tests/` to use `ainpp`.
- [ ] Create `tests/conftest.py` with shared fixtures for small dataset and dummy model.
- [ ] Configure `pytest`, `mypy`, and `flake8` in `pyproject.toml`.
- [ ] Add a `check-all` command to the project (e.g., via a simple script or `make`).
- [ ] Verify all tests pass with the new project structure.
- [ ] Ensure `mypy` passes for all core library modules.

## Dev Notes
- Use `pytest.mark.parametrize` where applicable for metric tests.
- Reference `STORY-001` for the final module structure.
