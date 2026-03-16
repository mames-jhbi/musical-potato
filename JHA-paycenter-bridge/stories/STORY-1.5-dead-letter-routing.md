# STORY-1.5 – Dead Letter Routing for Invalid Messages

**Epic:** [EPIC-1 – Kafka Consumer Service](../epics/EPIC-1-kafka-consumer.md)
**Status:** 🔵 Backlog | **Points:** 3

## User Story

> As a platform engineer, I want invalid or unparseable messages to be routed to a dead letter topic so that processing errors are preserved for investigation and replay without blocking the consumer.

## Details

- Route to `payment-events-dead-letter` Pub/Sub topic on:
  - JSON parse failure
  - Pydantic validation failure
  - Pub/Sub publish failure after max retries
- Dead letter message body: original raw Kafka bytes (base64 encoded if non-UTF-8)
- Dead letter message attributes:
  - `error_type`: `JSON_PARSE_ERROR` | `SCHEMA_VALIDATION_ERROR` | `PUBLISH_FAILURE`
  - `error_message`: exception message (truncated to 512 chars)
  - `kafka_topic`, `kafka_partition`, `kafka_offset`
  - `failed_at`: ISO 8601 timestamp

## Acceptance Criteria

- [ ] Malformed JSON is routed to DLT with `error_type = JSON_PARSE_ERROR`
- [ ] Failed schema validation is routed to DLT with `error_type = SCHEMA_VALIDATION_ERROR`
- [ ] Service continues consuming after routing to DLT — does not crash
- [ ] Dead letter message contains original raw bytes
- [ ] Unit tests cover each error type
