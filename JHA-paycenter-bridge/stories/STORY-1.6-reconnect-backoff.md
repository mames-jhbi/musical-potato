# STORY-1.6 – Broker Disconnect and Reconnect with Backoff

**Epic:** [EPIC-1 – Kafka Consumer Service](../epics/EPIC-1-kafka-consumer.md)
**Status:** 🔵 Backlog | **Points:** 2

## User Story

> As a platform engineer, I want the service to automatically reconnect to the Kafka broker after a disconnect so that transient network issues do not require a manual restart.

## Details

- On `KafkaException` or poll timeout, retry with exponential backoff: 1s → 2s → 4s → ... → 60s max
- Log each reconnect attempt with attempt count and wait duration
- After 10 consecutive failures, emit a CRITICAL log and continue retrying (do not exit)
- Expose reconnect attempt count as a metric `kafka_reconnect_attempts_total`

## Acceptance Criteria

- [ ] Service retries connection after simulated broker disconnect
- [ ] Backoff caps at 60 seconds maximum
- [ ] Reconnect attempts are logged with structured fields
- [ ] Service resumes consuming after broker comes back online
- [ ] Unit test mocks KafkaException and verifies backoff timing
