# STORY-2.4 – Dead Letter Topic Publishing

**Epic:** [EPIC-2 – GCP Pub/Sub Integration](../epics/EPIC-2-pubsub-integration.md)
**Status:** 🔵 Backlog | **Points:** 2

## User Story

> As a platform engineer, I want unparseable or undeliverable messages published to a dead letter topic so that no event is silently lost and we can replay or investigate failures.

## Details

- Dead letter topic: configurable via `PUBSUB_DLT_TOPIC` env var (default: `payment-events-dead-letter`)
- Dead letter message body: original raw Kafka bytes (base64 if binary)
- Attributes: `error_type`, `error_message`, `kafka_topic`, `kafka_partition`, `kafka_offset`, `failed_at`
- Dead letter publisher uses the same `PublisherClient` with no batching (immediate flush)

## Acceptance Criteria

- [ ] Messages failing validation are published to DLT within 1 second
- [ ] DLT message body contains the original raw bytes
- [ ] DLT message attributes contain all six error context fields
- [ ] A failure to publish to DLT is logged as CRITICAL but does not crash the service
- [ ] Unit tests assert DLT publish is called with correct attributes for each error type
