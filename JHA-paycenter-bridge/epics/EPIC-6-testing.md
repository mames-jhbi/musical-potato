# EPIC-6 – Testing & Quality Assurance

| Field | Value |
|-------|-------|
| **Epic ID** | EPIC-6 |
| **Status** | 🔵 Backlog |
| **Priority** | P1 |
| **Depends On** | EPIC-1, EPIC-2 |
| **Estimate** | Ongoing (parallel with EPIC-1 through EPIC-5) |

## Summary

Define and implement the test strategy covering unit tests, integration tests with a local Kafka instance, and end-to-end staging validation — with a minimum 80% code coverage gate on CI.

## Goal

Every pull request is validated against a full unit + integration test suite before merge. Staging environment tests confirm real connectivity to JH EES and at least one confirmed end-to-end message flow per rail type before production cutover.

## Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-6.1 | Unit tests SHALL cover: JSON schema validation, Pub/Sub attribute mapping, dead letter routing, health check endpoint, rail type filtering |
| FR-6.2 | Integration tests SHALL use a local Kafka instance (Testcontainers or docker-compose) to validate the full consume → publish flow without live GCP dependencies |
| FR-6.3 | A staging environment test SHALL validate connectivity to the JH EES Kafka cluster using synthetic test events from the JH integration team |
| FR-6.4 | Test coverage SHALL be measured and SHALL be no less than 80% for core application logic |

## Test Matrix

| Test Type | Scope | Tooling | Runs In |
|-----------|-------|---------|---------|
| Unit | Schema validation, filtering, attribute mapping | `pytest` + `unittest.mock` | CI (every PR) |
| Integration | Full Kafka → Pub/Sub flow | `pytest` + Testcontainers | CI (every PR) |
| Contract | Event 800 schema compliance | `jsonschema` + sample payloads from JH | CI (every PR) |
| Staging E2E | Live JH EES → GCP Pub/Sub | Manual + scripted | Pre-production gate |
| Load | Consumer throughput at peak TPS | `locust` or `k6` | On-demand |

## Acceptance Criteria

- [ ] `pytest` passes with ≥80% coverage on CI for every pull request
- [ ] Integration test suite passes against local Kafka + mock Pub/Sub
- [ ] Staging connectivity test confirms at least one end-to-end message per rail type (Zelle, RTP, FedNow)
- [ ] No unit tests use `time.sleep` — all async operations are properly mocked
- [ ] CI fails fast on coverage drop below threshold

## Stories

| Story | Title |
|-------|-------|
| [STORY-6.1](../stories/STORY-6.1-unit-tests.md) | Unit test suite for core logic |
| [STORY-6.2](../stories/STORY-6.2-integration-tests.md) | Integration tests with local Kafka (Testcontainers) |
| [STORY-6.3](../stories/STORY-6.3-contract-tests.md) | Contract tests for Event 800 schema |
| [STORY-6.4](../stories/STORY-6.4-staging-e2e.md) | Staging end-to-end validation |
| [STORY-6.5](../stories/STORY-6.5-ci-coverage-gate.md) | CI coverage gate (≥80%) |
