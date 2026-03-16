# STORY-4.1 – Terraform Module: Pub/Sub Topics

**Epic:** [EPIC-4 – Terraform IaC](../epics/EPIC-4-terraform-iac.md)
**Status:** 🔵 Backlog | **Points:** 2

## User Story

> As a platform engineer, I want the Pub/Sub topics provisioned via Terraform so that topic configuration is version-controlled and reproducible across environments.

## Details

Resources:
- `google_pubsub_topic.payment_events` — retention: 7 days, message ordering enabled
- `google_pubsub_topic.payment_events_dead_letter` — retention: 14 days

Labels: `environment`, `project`, `team`

## Acceptance Criteria

- [ ] Both topics created by `terraform apply`
- [ ] Message retention set correctly per topic
- [ ] `terraform plan` is clean after apply
- [ ] Topics created in the correct GCP project and region
