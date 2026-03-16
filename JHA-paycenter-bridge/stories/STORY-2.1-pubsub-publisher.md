# STORY-2.1 – Pub/Sub Publisher Client Setup and Batch Config

**Epic:** [EPIC-2 – GCP Pub/Sub Integration](../epics/EPIC-2-pubsub-integration.md)
**Status:** 🔵 Backlog | **Points:** 3

## User Story

> As a platform engineer, I want a configured Pub/Sub publisher client with optimised batch settings so that payment events are published efficiently with low latency.

## Details

- Use `google-cloud-pubsub` `PublisherClient`
- Batch settings: `max_messages=100`, `max_latency=0.1` seconds
- Topic: configurable via env var `PUBSUB_TOPIC` (default: `payment-events`)
- Authenticate via Workload Identity (Cloud Run) — no service account key file needed
- Publisher initialised once at startup and reused across messages

## Acceptance Criteria

- [ ] Publisher client initialises successfully on Cloud Run using Workload Identity
- [ ] Batch settings are applied: `max_messages=100`, `max_latency=0.1s`
- [ ] Topic name is read from `PUBSUB_TOPIC` environment variable
- [ ] Unit test mocks `PublisherClient` and verifies correct topic and settings
