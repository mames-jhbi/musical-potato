# STORY-6.2 – Integration Tests with Local Kafka (Testcontainers)

**Epic:** [EPIC-6 – Testing & Quality Assurance](../epics/EPIC-6-testing.md)
**Status:** 🔵 Backlog | **Points:** 5

## User Story

> As a platform engineer, I want integration tests that run against a real local Kafka instance so that I can validate the full consume → publish flow without depending on live GCP or JH environments.

## Details

- Use `testcontainers-python` to spin up a local Kafka container per test session
- Mock GCP Pub/Sub using `unittest.mock` or `pytest-mock` (avoid live GCP in CI)
- Test the full pipeline: produce a test message to local Kafka → consumer picks it up → (mocked) Pub/Sub publish is called with correct payload and attributes

Test scenarios:
1. Happy path — Zelle, RTP, FedNow event each published to (mock) Pub/Sub
2. Invalid message — routed to (mock) DLT
3. Disallowed rail type — consumed and discarded
4. Pub/Sub publish failure — retried 5 times, then DLT
5. Kafka consumer lag — consumer processes backlog of 50 messages

## Acceptance Criteria

- [ ] All integration tests pass in CI without a live Kafka or GCP connection
- [ ] Testcontainers starts and stops cleanly within the test session
- [ ] Tests complete within 60 seconds on CI
- [ ] Integration tests are in `tests/integration/` and excluded from unit test run
