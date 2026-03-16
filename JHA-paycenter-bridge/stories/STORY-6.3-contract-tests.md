# STORY-6.3 – Contract Tests for Event 800 Schema

**Epic:** [EPIC-6 – Testing & Quality Assurance](../epics/EPIC-6-testing.md)
**Status:** 🔵 Backlog | **Points:** 3

## User Story

> As a platform engineer, I want contract tests that validate real JH sample payloads against our schema model so that any breaking schema change from JH is detected immediately.

## Details

- Sample payloads provided by JH integration team (sanitised, per rail type and status)
- Store payloads in `tests/fixtures/event800/` as `.json` files
- Validate each fixture against the `Event800` Pydantic model
- If JH provides a JSON Schema or Avro schema, validate against that too

Test matrix (once JH provides samples):

| Fixture | Rail | Status |
|---------|------|--------|
| `zelle_completed.json` | ZELLE | COMPLETED |
| `zelle_failed.json` | ZELLE | FAILED |
| `rtp_completed.json` | RTP | COMPLETED |
| `rtp_returned.json` | RTP | RETURNED |
| `fednow_completed.json` | FEDNOW | COMPLETED |
| `fednow_failed.json` | FEDNOW | FAILED |

## Acceptance Criteria

- [ ] All JH-provided sample fixtures pass validation without errors
- [ ] Tests are blocked/skipped if fixture files are not yet present (with clear skip message)
- [ ] A fixture with a breaking schema change causes the test to fail with a clear diff
- [ ] Contract tests run in CI alongside unit tests

## Notes

- Blocked on JH providing sample payloads (Appendix A Q3)
