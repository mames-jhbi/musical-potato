# STORY-5.1 – Structured JSON Logging

**Epic:** [EPIC-5 – Observability & Operations](../epics/EPIC-5-observability.md)
**Status:** 🔵 Backlog | **Points:** 2

## User Story

> As an on-call engineer, I want every processed message to emit a structured log entry so that I can trace any payment from Kafka offset to Pub/Sub message ID in Cloud Logging without SSH access.

## Details

- Use Python `structlog` or `logging` with a JSON formatter
- Log level configurable via `LOG_LEVEL` env var (default: `INFO`)
- Every consumed message logs at INFO: `eventId`, `pmtRailType`, `paymentStatus`, `kafka_offset`, `pubsub_message_id`, `processing_duration_ms`
- Dead lettered messages log at WARNING with `error_type` and `error_message`
- Startup and shutdown events log at INFO

## Acceptance Criteria

- [ ] All log output is valid JSON (parseable by Cloud Logging)
- [ ] Every published message has a corresponding INFO log with all required fields
- [ ] Dead lettered messages have a WARNING log with error context
- [ ] Log level can be changed via env var without code change
- [ ] No plain-text log lines in production (no `print()` statements)
