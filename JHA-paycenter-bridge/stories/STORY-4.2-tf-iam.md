# STORY-4.2 – Terraform Module: IAM Service Account and Roles

**Epic:** [EPIC-4 – Terraform IaC](../epics/EPIC-4-terraform-iac.md)
**Status:** 🔵 Backlog | **Points:** 2

## User Story

> As a security engineer, I want the bridge service to run with a least-privilege service account provisioned via Terraform so that its blast radius is minimised in the event of a compromise.

## Details

Resources:
- `google_service_account.paycenter_bridge_sa`
- IAM bindings (minimum required only):
  - `roles/pubsub.publisher` on `payment-events` topic
  - `roles/pubsub.publisher` on `payment-events-dead-letter` topic
  - `roles/secretmanager.secretAccessor` on bridge secrets
  - `roles/run.invoker` (if health check invocation is needed)

## Acceptance Criteria

- [ ] Service account created with correct display name and description
- [ ] Exactly the minimum required IAM roles bound — no project-level bindings
- [ ] `terraform plan` shows zero drift after apply
- [ ] Service account email output available for use in Cloud Run and other modules
