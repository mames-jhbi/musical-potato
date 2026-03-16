# STORY-4.4 – Terraform Module: Cloud Run Service

**Epic:** [EPIC-4 – Terraform IaC](../epics/EPIC-4-terraform-iac.md)
**Status:** 🔵 Backlog | **Points:** 3

## User Story

> As a platform engineer, I want the Cloud Run service fully provisioned by Terraform so that service configuration is reproducible and environment-specific settings are managed via tfvars.

## Details

Resource: `google_cloud_run_v2_service.paycenter_bridge`

Key config:
- Image: `us-central1-docker.pkg.dev/<project>/paycenter-bridge/bridge:<tag>` (var)
- Min instances: 1, Max instances: 3
- CPU always allocated, not throttled
- Secret Manager volume mounts for all Kafka credentials
- Environment variables: `KAFKA_TOPIC`, `KAFKA_CONSUMER_GROUP_ID`, `PUBSUB_TOPIC`, `PUBSUB_DLT_TOPIC`, `LOG_LEVEL`, `ALLOWED_RAIL_TYPES`

## Acceptance Criteria

- [ ] Cloud Run service created and running after `terraform apply`
- [ ] Image tag is a Terraform variable (not hardcoded)
- [ ] Secret mounts reference the secrets from STORY-4.3
- [ ] Service uses the service account from STORY-4.2
- [ ] `terraform plan` is clean after apply
