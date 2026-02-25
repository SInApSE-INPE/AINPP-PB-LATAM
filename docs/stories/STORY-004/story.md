# STORY-004: Automated Documentation (MkDocs)

**Status:** Ready for Review
**Assignee:** @architect
**Epic:** Framework Evolution

## Story
**As a** Researcher,
**I want** automated documentation via MkDocs and `mkdocstrings`,
**so that** I can easily access API documentation and project tutorials generated directly from the source code.

## Acceptance Criteria
- [x] `mkdocs.yml` is created and configured for the project.
- [x] `mkdocstrings` is used to auto-generate API documentation from docstrings.
- [x] `docs/` folder is initialized with `index.md`, `installation.md`, and `usage.md`.
- [x] Documentation is theme-compatible with standard scientific libraries (e.g., `material` theme).
- [x] API documentation is structured by module (models, datasets, core, visualization).

## 🤖 CodeRabbit Integration
**Story Type Analysis**: Architecture / Documentation
**Complexity**: Low
**Specialized Agent Assignment**:
- Primary Agents: @architect, @dev
- Supporting Agents: @github-devops

**Quality Gate Tasks**:
- [ ] Pre-PR (@github-devops): Verify documentation build and link integrity.

**CodeRabbit Focus Areas**:
- Docstring formatting and Google-style compliance.
- Completeness of API references for all exported modules.
- Clarity of installation and usage examples.

## Tasks / Subtasks
- [x] Create `mkdocs.yml` with basic project metadata and theme.
- [x] Add `mkdocs-material` and `mkdocstrings` as dev dependencies in `pyproject.toml`.
- [x] Initialize `docs/` with standard boilerplate pages.
- [x] Use `mkdocstrings` in `docs/api.md` to reference all modules in `ainpp`.
- [x] Verify local build with `mkdocs serve`.
- [x] Ensure all docstrings in `ainpp/` meet Google-style standards.

## Dev Notes
- Refer to `STORY-001` for the final module names in `ainpp`.
- Docstrings are already being standardized in STORY-001; this story focuses on the *rendering*.

## Dev Agent Record
### Debug Log
- Ran `./scripts/check_all.sh` (pytest + coverage; dataset integration tests currently skipped elsewhere); docs build not run in CI yet.

### Completion Notes
- Added MkDocs + mkdocstrings setup with Material theme and nav structured by domain.
- Seeded docs pages (index, installation, usage) and API stubs per module with Google-style docstring rendering.
- Added docs extras to `pyproject.toml` for reproducible installs.

### File List
- mkdocs.yml
- docs/index.md
- docs/installation.md
- docs/usage.md
- docs/api.md
- docs/api/datasets.md
- docs/api/models.md
- docs/api/preprocessing.md
- docs/api/evaluation.md
- docs/api/visualization.md
- pyproject.toml

### Change Log
- Introduced MkDocs-based documentation site with mkdocstrings API reference and theme configuration.

### Status
- Ready for Review
