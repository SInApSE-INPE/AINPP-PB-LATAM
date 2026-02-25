# Spec: Refactor to Modern Scientific Library Structure

**Story ID:** STORY-001
**Status:** Approved
**Created By:** @architect (Aria)

---

## 1. Goal
Modernize the code structure into a standard scientific Python package named `ainpp`.

---

## 2. Dependencies
- `setuptools>=61.0`
- `hydra-core`
- `zarr`
- `xarray`
- `torch`

---

## 3. Files to Create/Modify
- **Create:** `pyproject.toml`
- **Modify:** `src/` -> `src/ainpp/` (Move and rename)
- **Modify:** All Python files (Fix imports)
- **Modify:** `src/ainpp/engine.py` (Refactor visualization)
- **Create:** `conf/dataset/gsmap.yaml`
- **Modify:** `save_dataset.py` -> `src/ainpp/preprocessing/pipeline.py` (Refactor to class-based)

---

## 4. Implementation Checklist
- [ ] Initialize `pyproject.toml` with standard build-system and project metadata.
- [ ] Migrate `src/` to `src/ainpp/` to support the SRC layout.
- [ ] Recursively update all imports from `src.` to `ainpp.` or relative paths.
- [ ] Remove all manual `sys.path.append` hacks in the codebase.
- [ ] Decouple `save_epoch_sample` from `run_training` in `engine.py`.
- [ ] Extract hardcoded GSMaP parameters into Hydra configuration files.
- [ ] Standardize type hints and docstrings in all core modules.

---

## 5. Testing Strategy
- [ ] Unit tests for all refactored modules.
- [ ] Integration test for the training loop with a small dataset.
- [ ] Verification of editable install via `pip install -e .`.
