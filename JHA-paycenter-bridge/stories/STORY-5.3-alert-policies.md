# STORY-5.3 – Cloud Monitoring Alert Policies

**Epic:** [EPIC-5 – Observability & Operations](../epics/EPIC-5-observability.md)
**Status:** 🔵 Backlog | **Points:** 2

## User Story

> As an on-call engineer, I want Cloud Monitoring alerts to fire when key thresholds are breached so that I'm notified before failures impact downstream consumers.

## Details

Provisioned via Terraform (STORY-4.5). This story covers testing and validation.

| Alert | Condition | Notification |
|-------|-----------|-------------|
| High consumer lag | lag > 1000 for 5 min | PagerDuty / email |
| Dead letter buildup | DLT count > 10 in 1 hour | Email |
| High error rate | Cloud Run error rate > 1% for 5 min | PagerDuty |
| Service down | Instance count = 0 for 60 sec | PagerDuty |

## Acceptance Criteria

- [ ] Each alert fires correctly when its threshold is breached in staging
- [ ] Alert notifications reach the configured channel within 2 minutes
- [ ] Alerts auto-resolve when the condition clears
- [ ] No false positives observed during normal operation over a 24-hour period
