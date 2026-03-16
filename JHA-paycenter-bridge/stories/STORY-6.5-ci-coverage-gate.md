# STORY-6.5 – CI Coverage Gate (≥80%)

**Epic:** [EPIC-6 – Testing & Quality Assurance](../epics/EPIC-6-testing.md)
**Status:** 🔵 Backlog | **Points:** 1

## User Story

> As a platform engineer, I want the CI pipeline to fail if test coverage drops below 80% so that coverage is a hard requirement, not a suggestion.

## Details

- `pytest-cov` with `--cov-fail-under=80`
- Coverage report uploaded as CI artifact on every run
- Coverage badge in README reflecting current `main` branch coverage
- Exclusions: `tests/`, `__init__.py`, migration scripts

## Acceptance Criteria

- [ ] CI fails on a PR that drops coverage below 80%
- [ ] Coverage report is available as a downloadable CI artifact
- [ ] Coverage badge in README is updated automatically on merge to `main`
- [ ] Exclusion list is documented in `pyproject.toml` or `setup.cfg`
