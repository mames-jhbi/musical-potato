# STORY-3.4 – Cloud Run Service Configuration

**Epic:** [EPIC-3 – Cloud Run Deployment](../epics/EPIC-3-cloud-run-deployment.md)
**Status:** 🔵 Backlog | **Points:** 3

## User Story

> As a platform engineer, I want the Cloud Run service configured with always-on CPU and a minimum of 1 instance so that the Kafka consumer group membership is stable and events are never missed during idle periods.

## Details

| Setting | Value |
|---------|-------|
| `min-instances` | 1 |
| `max-instances` | 3 |
| `cpu` | 1 |
| `memory` | 512Mi |
| `cpu-throttling` | disabled |
| `port` | 8080 |
| `timeout` | 3600s |
| `service-account` | `paycenter-bridge-sa@<project>.iam.gserviceaccount.com` |

## Acceptance Criteria

- [ ] Cloud Run service deploys without errors
- [ ] Min instance count of 1 means at least one instance is always running
- [ ] CPU is not throttled between requests
- [ ] Service account has only `pubsub.publisher` and `secretmanager.secretAccessor` roles
- [ ] Environment variables do not contain any secret values

## Notes

- Fully provisioned via Terraform (STORY-4.4) — this story covers the spec and manual validation
