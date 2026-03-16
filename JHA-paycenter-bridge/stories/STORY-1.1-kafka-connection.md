# STORY-1.1 – Kafka Connection with SASL/SSL Auth

**Epic:** [EPIC-1 – Kafka Consumer Service](../epics/EPIC-1-kafka-consumer.md)
**Status:** 🔵 Backlog | **Points:** 3

## User Story

> As a platform engineer, I want the bridge service to securely authenticate to the JH EES Kafka cluster at startup so that we can consume payment events without storing credentials in the codebase.

## Details

- Use `confluent-kafka-python` (`confluent_kafka.Consumer`)
- Auth config sourced from GCP Secret Manager (broker URL, username, password, SSL cert paths)
- Support `SASL_SSL` with either `PLAIN` or `SCRAM-SHA-512` (to be confirmed with JH — see Appendix A Q7)
- If mTLS is required, mount client cert and key from Secret Manager volume

## Acceptance Criteria

- [ ] Service connects to Kafka on startup using credentials from Secret Manager
- [ ] Connection failure at startup logs a structured error and retries with backoff
- [ ] Credentials are never logged or written to disk beyond Secret Manager mounts
- [ ] Unit test mocks the Kafka client and asserts correct config is passed

## Notes

- Blocked on JH integration questions: broker hostname, port, auth mechanism (Appendix A Q6, Q7)
