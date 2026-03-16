# STORY-2.2 – Message Attribute Mapping from Event 800

**Epic:** [EPIC-2 – GCP Pub/Sub Integration](../epics/EPIC-2-pubsub-integration.md)
**Status:** 🔵 Backlog | **Points:** 2

## User Story

> As a downstream consumer, I want each Pub/Sub message to carry rail type, status, and trace attributes so that I can filter and route messages without deserializing the full JSON payload.

## Details

Map the following attributes from the Event 800 payload:

| Attribute | Source Field | Notes |
|-----------|-------------|-------|
| `rail_type` | `pmtRailType` | |
| `payment_status` | `paymentStatus` | |
| `institution_id` | `institutionId` | |
| `event_id` | `eventId` | |
| `source_topic` | — | Kafka topic name, set by bridge |
| `kafka_offset` | — | Partition offset as string, set by bridge |

## Acceptance Criteria

- [ ] All six attributes present on every published Pub/Sub message
- [ ] Attribute values match the source Event 800 fields exactly
- [ ] `kafka_offset` is the string representation of the Kafka partition offset
- [ ] Unit tests assert all attributes for each rail type
