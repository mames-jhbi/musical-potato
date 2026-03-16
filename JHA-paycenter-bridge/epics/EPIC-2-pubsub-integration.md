# EPIC-2 – GCP Pub/Sub Integration

| Field | Value |
|-------|-------|
| **Epic ID** | EPIC-2 |
| **Status** | 🔵 Backlog |
| **Priority** | P0 – Critical Path |
| **Depends On** | EPIC-1 (Kafka Consumer Service) |
| **Estimate** | 1 sprint |

## Summary

Configure GCP Pub/Sub resources and implement the publishing logic, including message attributes for downstream filtering, ordering keys, and dead letter handling.

## Goal

Publish all valid Zelle, RTP, and FedNow Event 800 messages to a single unified GCP Pub/Sub topic (`payment-events`) with rich message attributes, enabling downstream subscribers to filter and route without deserializing the full payload.

## Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-2.1 | The service SHALL publish each valid Event 800 message to the `payment-events` Pub/Sub topic with the raw JSON payload as the message body |
| FR-2.2 | The service SHALL attach message attributes: `rail_type`, `payment_status`, `institution_id`, `event_id`, `source_topic`, `kafka_offset` |
| FR-2.3 | A separate Pub/Sub topic `payment-events-dead-letter` SHALL receive messages that cannot be parsed or published |
| FR-2.4 | The dead letter message SHALL include: original raw Kafka bytes, topic/partition/offset, error type, error message, and timestamp |
| FR-2.5 | Pub/Sub message ordering keys SHALL be set to `transactionId` where available |

## Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-2.1 | The publisher SHALL use batch settings: `max_messages=100`, `max_latency=0.1s` |
| NFR-2.2 | The service SHALL retry Pub/Sub publish failures up to 5 times with exponential backoff before routing to dead letter |
| NFR-2.3 | Published messages SHALL log the Pub/Sub message ID alongside the Kafka offset for end-to-end traceability |

## Message Attributes

| Attribute | Example | Source |
|-----------|---------|--------|
| `rail_type` | `ZELLE` | `pmtRailType` field in Event 800 |
| `payment_status` | `COMPLETED` | `paymentStatus` field in Event 800 |
| `institution_id` | `FI-00123` | `institutionId` field in Event 800 |
| `event_id` | `uuid` | `eventId` field in Event 800 |
| `source_topic` | `ees-paycenter-800` | Kafka topic name (set by bridge) |
| `kafka_offset` | `1048293` | Kafka partition offset (set by bridge) |

## Acceptance Criteria

- [ ] `payment-events` Pub/Sub topic receives messages with correct attributes for all three rail types
- [ ] A downstream test subscriber filtering on `rail_type = ZELLE` receives only Zelle events
- [ ] A simulated publish failure (IAM permission revocation) routes the message to `payment-events-dead-letter` after retries are exhausted
- [ ] Dead letter messages contain the original raw payload and full error context
- [ ] Pub/Sub message IDs are logged alongside Kafka offsets

## Stories

| Story | Title |
|-------|-------|
| [STORY-2.1](../stories/STORY-2.1-pubsub-publisher.md) | Pub/Sub publisher client setup and batch config |
| [STORY-2.2](../stories/STORY-2.2-message-attributes.md) | Message attribute mapping from Event 800 |
| [STORY-2.3](../stories/STORY-2.3-ordering-keys.md) | Ordering key configuration by transactionId |
| [STORY-2.4](../stories/STORY-2.4-dead-letter-publisher.md) | Dead letter topic publishing |
| [STORY-2.5](../stories/STORY-2.5-publish-retry.md) | Publish retry logic with exponential backoff |
