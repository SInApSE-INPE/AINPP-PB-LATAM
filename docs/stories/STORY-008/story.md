# STORY-008: Refresh README for Unified CLI & Docs

**Status:** Ready for Review
**Assignee:** @dev
**Epic:** Framework Evolution

## Story
**As a** maintainer,  
**I want** the README to reflect the unified Hydra CLI, testing workflow, and documentation site,  
**so that** contributors and users can quickly understand how to install, run, test, and view docs.

## Acceptance Criteria
- [ ] README includes:
  - [ ] Table of content.
  - [ ] Quick start installation using `pip install -e .[dev,docs]`.
  - [ ] Usage examples for `main.py task=preprocess|train|evaluate` with Hydra overrides.
    - [ ] Describe and show example with all deep learning models. 
  - [ ] How to run quality gates: `./scripts/check_all.sh`.
  - [ ] Link to the MkDocs site (GitHub Pages) and local preview instructions.
  - [ ] Badges or status placeholders for tests/docs (if available).
- [ ] README references current directory structure (from STORY-001/002/005/006/004).
- [ ] No stale references to legacy scripts (`scripts/preprocess.py`, etc.) without noting deprecation.
- [ ] Add classical badges should be included in the header of the README
- [ ] Markdown passes basic lint (no broken links, code fences valid).

## 🤖 CodeRabbit Integration
**Story Type Analysis**: Docs / DevEx  
**Complexity**: Low  
**Specialized Agent Assignment**: Primary @dev, Supporting @architect  

**Quality Gate Tasks**:
- [ ] Pre-PR (@github-devops): Verify links and code fences render correctly on GitHub.

**CodeRabbit Focus Areas**:
- Accuracy of commands and paths.
- Alignment with Hydra-based CLI and testing workflow.
- Removal of outdated references.

## Tasks / Subtasks
- [ ] Update README content per acceptance criteria.
- [ ] Add/refresh badges (tests/docs) if available.
- [ ] Validate Markdown rendering locally (e.g., `markdownlint` or GitHub preview).

## Dev Notes
- Pull commands and paths from existing stories: STORY-002 (CLI), STORY-004 (docs), STORY-005/006 (tests/coverage), STORY-001 (structure).
- Keep README concise; link to docs for deeper detail.
