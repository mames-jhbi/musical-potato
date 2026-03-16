# EPIC-5 – Observability & Operations

| Field | Value |
|-------|-------|
| **Epic ID** | EPIC-5 |
| **Status** | 🔵 Backlog |
| **Priority** | P1 |
| **Depends On** | EPIC-1, EPIC-2, EPIC-3 |
| **Estimate** | 1 sprint |

## Summary

Implement structured logging, custom metrics, alerting, and a Cloud Monitoring dashboard to ensure the bridge service is fully observable in production from day one.

## Goal

Any failure mode — consumer lag, dead letter buildup, publish errors, or service outage — is detected and alerted within 5 minutes, with enough log context to diagnose root cause without SSH access.

## Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-5.1 | Emit structured JSON logs to stdout captured by Cloud Logging. Each entry SHALL include: `timestamp`, `severity`, `eventId`, `pmtRailType`, `paymentStatus`, `kafka_offset`, `pubsub_message_id`, `processing_duration_ms` |
| FR-5.2 | Emit custom metrics via OpenTelemetry SDK: `messages_consumed_total` (by `rail_type`), `messages_published_total` (by `rail_type`), `messages_dead_lettered_total`, `publish_latency_ms` (histogram), `kafka_consumer_lag` |
| FR-5.3 | Configure Cloud Monitoring alert policies for: consumer lag > 1000 messages for 5 min, dead letter count > 10 in 1 hour, Cloud Run error rate > 1% over 5 min, Cloud Run instance count = 0 for > 60 seconds |
| FR-5.4 | Create a Cloud Monitoring dashboard displaying all key metrics in a single view |

## Alert Policies

| Alert | Condition | Severity |
|-------|-----------|----------|
| High consumer lag | `kafka_consumer_lag > 1000` for 5 min | 🔴 Critical |
| Dead letter buildup | DLT message count > 10 in 1 hour | 🟠 Warning |
| High error rate | Cloud Run error rate > 1% over 5 min | 🔴 Critical |
| Service down | Cloud Run instance count = 0 for > 60s | 🔴 Critical |

## Acceptance Criteria

- [ ] Cloud Logging shows structured log entries for every consumed and published message
- [ ] Cloud Monitoring dashboard displays all five metric series with live data
- [ ] Injecting 15 dead letter messages triggers a Cloud Monitoring alert within 5 minutes
- [ ] Log entries include enough context to trace a payment from Kafka offset to Pub/Sub message ID

## Stories

| Story | Title |
|-------|-------|
| [STORY-5.1](../stories/STORY-5.1-structured-logging.md) | Structured JSON logging |
| [STORY-5.2](../stories/STORY-5.2-otel-metrics.md) | OpenTelemetry custom metrics |
| [STORY-5.3](../stories/STORY-5.3-alert-policies.md) | Cloud Monitoring alert policies |
| [STORY-5.4](../stories/STORY-5.4-dashboard.md) | Cloud Monitoring dashboard |
