# STORY-1.4 – Rail Type Filtering (Zelle / RTP / FedNow)

**Epic:** [EPIC-1 – Kafka Consumer Service](../epics/EPIC-1-kafka-consumer.md)
**Status:** 🔵 Backlog | **Points:** 2

## User Story

> As a platform engineer, I want the bridge to pass through only Zelle, RTP, and FedNow events so that unrelated rail types on the same topic do not pollute downstream Pub/Sub consumers.

## Details

- Allowed rail types: `ZELLE`, `RTP`, `FEDNOW` (enum configurable via env var `ALLOWED_RAIL_TYPES`)
- Events with `pmtRailType` not in allowed set: acknowledge Kafka offset and discard silently
- Discarded events are counted in `messages_discarded_total` metric (by rail type)

## Acceptance Criteria

- [ ] `ZELLE`, `RTP`, `FEDNOW` events are forwarded to Pub/Sub
- [ ] An event with `pmtRailType = ACH` is consumed, acknowledged, and discarded without Pub/Sub publish
- [ ] Discarded events are logged at DEBUG level with `eventId` and `pmtRailType`
- [ ] Allowed rail types are configurable without code change (env var)
- [ ] Unit tests cover all three allowed types and at least two disallowed types
