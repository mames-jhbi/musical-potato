# STORY-4.3 – Terraform Module: Secret Manager Placeholders

**Epic:** [EPIC-4 – Terraform IaC](../epics/EPIC-4-terraform-iac.md)
**Status:** 🔵 Backlog | **Points:** 2

## User Story

> As a platform engineer, I want Secret Manager secrets provisioned as empty placeholders by Terraform so that the secret structure is version-controlled but actual values are never stored in state or source control.

## Details

Secrets to provision (empty, no initial version):
- `paycenter-bridge-kafka-username`
- `paycenter-bridge-kafka-password`
- `paycenter-bridge-kafka-ssl-ca-cert` (if mTLS — TBD)
- `paycenter-bridge-kafka-ssl-client-cert` (if mTLS — TBD)
- `paycenter-bridge-kafka-ssl-client-key` (if mTLS — TBD)

Secret values populated out-of-band by ops team. Never in `.tfvars` or Terraform state.

## Acceptance Criteria

- [ ] All secret resources created by `terraform apply`
- [ ] No `secret_data` or initial version in Terraform code
- [ ] Secrets accessible by bridge service account (IAM binding from STORY-4.2)
- [ ] `terraform destroy` removes secrets (with lifecycle protection in prod)
