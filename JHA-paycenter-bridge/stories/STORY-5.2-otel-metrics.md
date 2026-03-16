# STORY-5.2 – OpenTelemetry Custom Metrics

**Epic:** [EPIC-5 – Observability & Operations](../epics/EPIC-5-observability.md)
**Status:** 🔵 Backlog | **Points:** 3

## User Story

> As an on-call engineer, I want real-time metrics on message throughput, latency, and consumer lag in Cloud Monitoring so that I can detect processing issues before they cause data loss.

## Details

Metrics to emit via `opentelemetry-sdk` + `opentelemetry-exporter-gcp-monitoring`:

| Metric | Type | Labels |
|--------|------|--------|
| `messages_consumed_total` | Counter | `rail_type` |
| `messages_published_total` | Counter | `rail_type` |
| `messages_discarded_total` | Counter | `rail_type` |
| `messages_dead_lettered_total` | Counter | `error_type` |
| `publish_latency_ms` | Histogram | `rail_type` |
| `kafka_consumer_lag` | Gauge | — |
| `kafka_reconnect_attempts_total` | Counter | — |

Export interval: 60 seconds.

## Acceptance Criteria

- [ ] All metrics visible in Cloud Monitoring within 2 minutes of service start
- [ ] `publish_latency_ms` histogram shows p50, p95, p99 buckets
- [ ] `kafka_consumer_lag` reflects actual lag from Kafka `committed_offsets` vs `highwater`
- [ ] Metrics export does not block or slow the consumer loop
