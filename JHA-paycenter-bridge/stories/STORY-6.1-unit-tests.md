# STORY-6.1 – Unit Test Suite for Core Logic

**Epic:** [EPIC-6 – Testing & Quality Assurance](../epics/EPIC-6-testing.md)
**Status:** 🔵 Backlog | **Points:** 5

## User Story

> As a platform engineer, I want a comprehensive unit test suite covering all core logic so that regressions are caught before code reaches the dev environment.

## Details

- Framework: `pytest` with `unittest.mock`
- Coverage tool: `pytest-cov`
- Coverage threshold: ≥80% (enforced in CI)

Modules to test:

| Module | Test Scenarios |
|--------|---------------|
| `event_validator.py` | Valid payload, missing fields, wrong types, extra fields, invalid JSON |
| `rail_filter.py` | Each allowed rail type, each disallowed type, missing pmtRailType field |
| `attribute_mapper.py` | All six attributes for each rail type, missing optional fields |
| `dead_letter.py` | Each error type, DLT publish failure |
| `health.py` | Healthy state, Kafka disconnected, Pub/Sub unavailable |
| `retry.py` | Retryable errors (all 5 retries), non-retryable (immediate DLT), success on 3rd attempt |

## Acceptance Criteria

- [ ] `pytest --cov=src --cov-report=term-missing` shows ≥80% coverage
- [ ] All tests pass in CI on every PR
- [ ] No tests use `time.sleep` — time is mocked where needed
- [ ] Test file naming follows `test_<module>.py` convention
