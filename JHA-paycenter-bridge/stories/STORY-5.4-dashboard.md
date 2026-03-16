# STORY-5.4 – Cloud Monitoring Dashboard

**Epic:** [EPIC-5 – Observability & Operations](../epics/EPIC-5-observability.md)
**Status:** 🔵 Backlog | **Points:** 2

## User Story

> As an on-call engineer, I want a single Cloud Monitoring dashboard showing all key bridge metrics so that I can assess service health at a glance without writing ad-hoc queries.

## Details

Dashboard panels:
1. Messages consumed per minute (by rail type) — line chart
2. Messages published per minute (by rail type) — line chart
3. Dead letter messages — counter + sparkline
4. Publish latency p50 / p95 / p99 — line chart
5. Kafka consumer lag — gauge + line chart
6. Cloud Run instance count — line chart
7. Cloud Run error rate — line chart
8. Kafka reconnect attempts — counter

Provisioned via Terraform `google_monitoring_dashboard` resource.

## Acceptance Criteria

- [ ] Dashboard visible in Cloud Monitoring UI after `terraform apply`
- [ ] All 8 panels show live data within 2 minutes of service start
- [ ] Dashboard is accessible to the on-call team without additional IAM changes
- [ ] Dashboard JSON exported and committed to repository for version control
