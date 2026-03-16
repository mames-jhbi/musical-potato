# STORY-1.3 – Event 800 Deserialization and Schema Validation

**Epic:** [EPIC-1 – Kafka Consumer Service](../epics/EPIC-1-kafka-consumer.md)
**Status:** 🔵 Backlog | **Points:** 5

## User Story

> As a platform engineer, I want each consumed Kafka message to be deserialized and validated against the Event 800 schema so that only well-formed, complete payment events are forwarded to Pub/Sub.

## Details

- Deserialize message value as UTF-8 JSON
- Validate required fields: `eventId`, `eventType`, `eventDateTime`, `pmtRailType`, `transactionId`, `paymentStatus`, `amount`, `currency`
- Use `pydantic` v2 for schema definition and validation
- If Schema Registry is in use (TBD — Appendix A Q8), add Avro/JSON Schema deserialization
- Schema model should live in `src/models/event800.py`

## Acceptance Criteria

- [ ] Valid Event 800 JSON is parsed into a typed `Event800` Pydantic model
- [ ] Missing required fields raise a `ValidationError` and route to dead letter
- [ ] Unknown extra fields are ignored (non-strict mode)
- [ ] Invalid JSON (not parseable) is caught and routed to dead letter with raw bytes preserved
- [ ] Unit tests cover: valid payload, missing required field, invalid JSON, extra fields

## Notes

- Schema may differ by rail type — confirm with JH (Appendix A Q2)
- Schema Registry integration to be determined after JH response (Appendix A Q8)
