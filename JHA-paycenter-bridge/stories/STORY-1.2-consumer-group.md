# STORY-1.2 – Consumer Group and Offset Management

**Epic:** [EPIC-1 – Kafka Consumer Service](../epics/EPIC-1-kafka-consumer.md)
**Status:** 🔵 Backlog | **Points:** 3

## User Story

> As a platform engineer, I want the bridge service to use a named consumer group with manual offset commits so that we guarantee at-least-once delivery and can safely restart without losing or duplicating events.

## Details

- Consumer group ID: `paycenter-bridge-cg` (confirm naming convention with JH — Appendix A Q8)
- `enable.auto.commit = false` — offsets committed only after successful Pub/Sub publish
- On rebalance, in-flight messages are tracked and uncommitted offsets are not lost
- Consumer group name should be configurable via environment variable

## Acceptance Criteria

- [ ] Consumer group ID is read from environment variable `KAFKA_CONSUMER_GROUP_ID`
- [ ] Auto-commit is disabled; offsets committed manually after Pub/Sub ACK
- [ ] On service restart, consumption resumes from last committed offset
- [ ] Rebalance events are logged with partition assignment details
- [ ] Unit test verifies `commit()` is called after — and only after — successful publish

## Notes

- Confirm consumer group naming restrictions with JH (Appendix A Q8)
