# EPIC-4 – Terraform Infrastructure as Code

| Field | Value |
|-------|-------|
| **Epic ID** | EPIC-4 |
| **Status** | 🔵 Backlog |
| **Priority** | P1 |
| **Depends On** | EPIC-2, EPIC-3 (for resource definitions) |
| **Estimate** | 1–2 sprints |

## Summary

Provision all GCP infrastructure required for the bridge service using Terraform, with remote state, least-privilege IAM, modular structure, and support for dev and prod workspaces.

## Goal

All GCP resources — Pub/Sub topics, Cloud Run service, IAM service accounts, Secret Manager secrets, and monitoring — are fully reproducible via `terraform apply` with zero manual steps.

## Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-4.1 | Provision `payment-events` Pub/Sub topic with 7-day message retention |
| FR-4.2 | Provision `payment-events-dead-letter` Pub/Sub topic with 14-day retention |
| FR-4.3 | Provision Cloud Run service with all env vars, min/max instance counts, and Secret Manager mounts |
| FR-4.4 | Create a dedicated GCP Service Account with minimum required IAM roles: `roles/pubsub.publisher`, `roles/secretmanager.secretAccessor`, `roles/run.invoker` |
| FR-4.5 | Provision all required Secret Manager secrets as empty placeholders. Values populated out-of-band — never in Terraform state or source control |
| FR-4.6 | Provision Cloud Monitoring alert policies for the metrics defined in EPIC-5 |
| FR-4.7 | Terraform state SHALL be stored in a GCS backend bucket with versioning enabled |
| FR-4.8 | Codebase SHALL be organised into modules: `pubsub`, `cloud_run`, `iam`, `secrets`, `monitoring` |

## Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-4.1 | `terraform plan` SHALL produce zero changes on a freshly deployed environment |
| NFR-4.2 | All resources SHALL be tagged with `environment`, `project`, and `team` labels |
| NFR-4.3 | Configuration SHALL support at least two workspaces: `dev` and `prod` |

## Module Structure

```
terraform/
├── main.tf
├── variables.tf
├── outputs.tf
├── backend.tf
├── modules/
│   ├── pubsub/
│   ├── cloud_run/
│   ├── iam/
│   ├── secrets/
│   └── monitoring/
└── envs/
    ├── dev.tfvars
    └── prod.tfvars
```

## Acceptance Criteria

- [ ] `terraform apply` from scratch provisions all required resources without errors
- [ ] Service account has exactly the minimum required IAM roles — no overpermissioning
- [ ] Destroy and re-apply produces identical infrastructure
- [ ] No secret values in Terraform state or `.tfvars` files committed to source control
- [ ] `terraform plan` shows zero diff on a clean environment

## Stories

| Story | Title |
|-------|-------|
| [STORY-4.1](../stories/STORY-4.1-tf-pubsub.md) | Terraform module: Pub/Sub topics |
| [STORY-4.2](../stories/STORY-4.2-tf-iam.md) | Terraform module: IAM service account and roles |
| [STORY-4.3](../stories/STORY-4.3-tf-secrets.md) | Terraform module: Secret Manager placeholders |
| [STORY-4.4](../stories/STORY-4.4-tf-cloud-run.md) | Terraform module: Cloud Run service |
| [STORY-4.5](../stories/STORY-4.5-tf-monitoring.md) | Terraform module: Cloud Monitoring alerts |
| [STORY-4.6](../stories/STORY-4.6-tf-backend.md) | Terraform remote state backend (GCS) |
