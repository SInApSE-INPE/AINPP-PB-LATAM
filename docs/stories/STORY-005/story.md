# STORY-005: Test Refactor & Module Coverage

**Status:** In Progress
**Assignee:** @dev
**Epic:** Framework Evolution

## Story
**As a** maintainer,  
**I want** tests organized by module/feature with best-practice patterns,  
**so that** each component can be validated independently and regressions are caught early.

## Acceptance Criteria
- [x] Every core module (`ainpp_pb_latam.*`) has dedicated unit tests covering happy path and key edge cases.
- [x] Tests run isolated (no shared mutable state, temp dirs/fixtures clean up automatically).
- [x] Coverage threshold ≥ 80% overall and ≥ 70% per module enforced in CI.
- [x] Fixtures live in `tests/conftest.py` and are reused (no duplicated setup code).
- [x] Parametrized tests used for shape/metric/value checks where inputs vary.
- [x] Test names are descriptive and follow `test_<unit>__<behavior>` convention.
- [x] Tests are deterministic (no network, seeded randomness).
- [x] CI exposes a single command `npm run test` (or equivalent) that runs lint+typecheck+pytest+coverage.

## 🤖 CodeRabbit Integration
**Story Type Analysis**: QA / Architecture  
**Complexity**: Medium  
**Specialized Agent Assignment**:
- Primary Agents: @dev, @qa
- Supporting Agents: @github-devops

**Quality Gate Tasks**:
- [ ] Pre-Commit (@dev): Ensure coverage thresholds and isolated tests validated locally.

**CodeRabbit Focus Areas**:
- Test isolation and fixture hygiene.
- Coverage enforcement per-module.
- Deterministic seeding and reproducibility.

## Tasks / Subtasks
- [x] Audit existing tests; map gaps per module.
- [x] Add/organize fixtures in `tests/conftest.py` (data loaders, dummy models, temp dirs).
- [x] Refactor tests to use parametrization and descriptive naming.
- [x] Introduce coverage configuration (fail under thresholds) and integrate with CI command.
- [x] Remove or rewrite flaky / integration-only tests that break isolation; add mocks where needed.

## Dev Notes
- Prefer `pytest` fixtures + `tmp_path` for file outputs; avoid writing to repo paths.
- Seed RNG in tests (`numpy`, `torch`, `random`) for reproducibility.
- Keep integration tests behind markers (e.g., `@pytest.mark.integration`) if needed.

## Dev Agent Record
### Debug Log
- Tests not executed in this session; quality gates pending.

### Completion Notes
- Consolidated deterministic fixtures in `tests/conftest.py` and rewrote tests around real modules.
- Added coverage/mypy/formatting gates via pytest-cov and `scripts/check_all.sh` single command.
- Enforced per-file coverage ≥70% with `scripts/enforce_coverage.py`.

### File List
- pyproject.toml
- scripts/check_all.sh
- scripts/enforce_coverage.py
- tests/conftest.py
- tests/test_losses.py
- tests/test_utils.py
- tests/test_datasets.py
- tests/test_visualization.py

### Change Log
- Established modern testing workflow with fixtures, coverage thresholds, and unified check command.

### Status
- In Progress (awaiting test run and validation)
