# STORY-4.6 – Terraform Remote State Backend (GCS)

**Epic:** [EPIC-4 – Terraform IaC](../epics/EPIC-4-terraform-iac.md)
**Status:** 🔵 Backlog | **Points:** 1

## User Story

> As a platform engineer, I want Terraform state stored in GCS with versioning enabled so that state is never lost and we can roll back if needed.

## Details

```hcl
terraform {
  backend "gcs" {
    bucket  = "paycenter-bridge-tf-state"
    prefix  = "terraform/state"
  }
}
```

- GCS bucket must have versioning enabled and uniform bucket-level access
- Separate state prefix per workspace (`dev/`, `prod/`)
- Bucket provisioned separately (bootstrap step) — not managed by this Terraform config

## Acceptance Criteria

- [ ] `terraform init` successfully configures GCS backend
- [ ] State file written to GCS bucket after `terraform apply`
- [ ] GCS bucket has versioning enabled
- [ ] Dev and prod workspaces use separate state prefixes
