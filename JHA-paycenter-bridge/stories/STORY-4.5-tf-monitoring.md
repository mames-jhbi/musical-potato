# STORY-4.5 – Terraform Module: Cloud Monitoring Alerts

**Epic:** [EPIC-4 – Terraform IaC](../epics/EPIC-4-terraform-iac.md)
**Status:** 🔵 Backlog | **Points:** 3

## User Story

> As a platform engineer, I want Cloud Monitoring alert policies provisioned via Terraform so that production alerting is version-controlled and consistently applied across environments.

## Details

Alert policies to provision:

| Alert | Metric | Threshold | Duration |
|-------|--------|-----------|----------|
| High consumer lag | `kafka_consumer_lag` | > 1000 | 5 min |
| Dead letter buildup | DLT message count | > 10 | 1 hour |
| High error rate | Cloud Run error rate | > 1% | 5 min |
| Service down | Cloud Run instance count | = 0 | 60 sec |

Notification channel: configurable via variable (email, PagerDuty, Slack webhook).

## Acceptance Criteria

- [ ] All four alert policies created by `terraform apply`
- [ ] Notification channel is a Terraform variable (not hardcoded)
- [ ] Alerts fire correctly when thresholds are breached (validated in EPIC-5)
- [ ] `terraform plan` is clean after apply
