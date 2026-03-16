# EPIC-1 – Kafka Consumer Service

| Field | Value |
|-------|-------|
| **Epic ID** | EPIC-1 |
| **Status** | 🔵 Backlog |
| **Priority** | P0 – Critical Path |
| **Blocked By** | JH PayCenter integration questions (Appendix A) |
| **Estimate** | 2 sprints |

## Summary

Build the core Python service that connects to the Jack Henry EES Kafka topic, consumes Event 800 (Payment Hub Activity Notifications) messages for Zelle, RTP, and FedNow, and prepares them for publishing to GCP Pub/Sub.

## Goal

Enable reliable, real-time consumption of payment events from JH PayCenter's Enterprise Event System with at-least-once delivery semantics, proper error handling, and offset management.

## Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-1.1 | The service SHALL connect to the JH EES Kafka cluster using SASL/SSL authentication with credentials sourced from GCP Secret Manager at startup |
| FR-1.2 | The service SHALL join a named consumer group (e.g. `paycenter-bridge-cg`) to enable offset tracking and horizontal scaling |
| FR-1.3 | The service SHALL subscribe to the Event 800 Kafka topic and consume messages in real time |
| FR-1.4 | The service SHALL deserialize each Kafka message value as JSON and validate it against the expected Event 800 schema |
| FR-1.5 | The service SHALL filter consumed events to include only those with `pmtRailType` values of `ZELLE`, `RTP`, or `FEDNOW`. Events from other rail types SHALL be acknowledged and discarded |
| FR-1.6 | The service SHALL commit Kafka offsets only after a message has been successfully published to Pub/Sub (at-least-once delivery guarantee) |
| FR-1.7 | The service SHALL route messages that fail JSON parsing or schema validation to the Pub/Sub dead letter topic, with the original raw bytes preserved |

## Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1.1 | Consumer poll loop latency from Kafka receipt to Pub/Sub publish SHALL be under 2 seconds at p99 under normal load |
| NFR-1.2 | The service SHALL handle Kafka broker disconnects and reconnect automatically with exponential backoff (max 60 seconds) |
| NFR-1.3 | The service SHALL emit structured JSON logs for every consumed message, including `eventId`, `pmtRailType`, `paymentStatus`, and processing outcome |

## Acceptance Criteria

- [ ] Service starts, authenticates to Kafka, and begins consuming within 30 seconds of container boot
- [ ] A test Zelle, RTP, and FedNow Event 800 message injected into Kafka each appear in Pub/Sub within 2 seconds
- [ ] An Event 800 message with an unknown rail type is consumed, acknowledged, and discarded — not published to Pub/Sub
- [ ] A malformed JSON message is routed to the dead letter topic without crashing the service
- [ ] Kafka offsets are not committed until Pub/Sub publish is confirmed

## Stories

| Story | Title |
|-------|-------|
| [STORY-1.1](../stories/STORY-1.1-kafka-connection.md) | Kafka connection with SASL/SSL auth |
| [STORY-1.2](../stories/STORY-1.2-consumer-group.md) | Consumer group and offset management |
| [STORY-1.3](../stories/STORY-1.3-event-deserialization.md) | Event 800 deserialization and schema validation |
| [STORY-1.4](../stories/STORY-1.4-rail-type-filtering.md) | Rail type filtering (Zelle / RTP / FedNow) |
| [STORY-1.5](../stories/STORY-1.5-dead-letter-routing.md) | Dead letter routing for invalid messages |
| [STORY-1.6](../stories/STORY-1.6-reconnect-backoff.md) | Broker disconnect and reconnect with backoff |
