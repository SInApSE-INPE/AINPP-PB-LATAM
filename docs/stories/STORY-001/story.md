# STORY-001: Refactor to Modern Scientific Library Structure

**Status:** Ready for Review
**Assignee:** Dex (Builder)
**Epic:** Framework Evolution

## Description
Refactor the current codebase to adopt a standard scientific Python library structure, including proper packaging (`pyproject.toml`), modularity, and better documentation. This will improve usability, allow for editable installations, and prepare the project for future scaling.

## Requirements
1.  **Packaging:** Add `pyproject.toml` for standard dependency and build management.
2.  **Source Layout:** Rename `src/` to `src/ainpp_pb_latam/` and update all internal imports.
3.  **Engine Refactoring:** Decouple training logic from visualization and metrics.
4.  **Config Management:** Integrate Hydra for all dataset and preprocessing parameters.
5.  **Quality Standards:** Apply consistent type hints and Google-style docstrings.

## Tasks
- [x] Phase 1: Setup & Packaging
    - [x] 1.1 Create pyproject.toml
    - [x] 1.2 Migrate to src/ainpp_pb_latam/ layout
- [x] Phase 2: Core Refactoring
    - [x] 2.1 Update project imports
    - [x] 2.2 Decouple training engine from visualization
- [x] Phase 3: Data & Configuration
    - [x] 3.1 Create Hydra configuration for GSMaP
    - [x] 3.2 Refactor preprocessing pipeline
- [x] Phase 4: Documentation & Polish
    - [x] 4.1 Standardize docstrings and type hints

## Acceptance Criteria
- [x] `pip install -e .` works without errors.
- [x] `from ainpp_pb_latam.models import ...` works from any directory.
- [x] No manual `sys.path.append(str(ROOT))` in core modules.
- [x] Training loop runs with standardized configurations.
- [x] Documentation can be auto-generated from docstrings.

## Dev Agent Record

### Agent Model Used
Gemini 2.0 Flash Thinking

### Debug Log
- [2026-02-25] Started STORY-001. Initializing package structure.
- [2026-02-25] Created pyproject.toml and migrated to src/ainpp_pb_latam/ layout.
- [2026-02-25] Updated all imports to ainpp_pb_latam. namespace.
- [2026-02-25] Decoupled visualization from training engine.
- [2026-02-25] Refactored preprocessing into PreprocessingPipeline.
- [2026-02-25] Standardized docstrings and type hints in core modules.

### Completion Notes
- Package `ainpp-pb-latam` is now installable and follows standard SRC layout.
- Visualization logic moved to `ainpp_pb_latam.visualization.samples`.
- Preprocessing modularized and integrated with Hydra.
- All core imports updated and verified with `pip install -e .`.

### Change Log
- [2026-02-25] Updated story status to In Progress and added tasks.
- [2026-02-25] Completed all implementation tasks.
- [2026-02-25] Set status to Ready for Review.
