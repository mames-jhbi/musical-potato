# STORY-3.5 – CI/CD Pipeline for Build, Test, and Deploy

**Epic:** [EPIC-3 – Cloud Run Deployment](../epics/EPIC-3-cloud-run-deployment.md)
**Status:** 🔵 Backlog | **Points:** 5

## User Story

> As a platform engineer, I want a CI/CD pipeline that builds, tests, and deploys the bridge service automatically so that every merged PR reaches the dev environment without manual steps.

## Details

Pipeline stages:
1. **Lint** — `ruff` + `mypy`
2. **Test** — `pytest` with coverage gate (≥80%)
3. **Build** — `docker build` multi-stage
4. **Push** — push to Artifact Registry (`us-central1-docker.pkg.dev/<project>/paycenter-bridge/bridge:<sha>`)
5. **Deploy (dev)** — `gcloud run deploy` or `terraform apply` targeting dev workspace
6. **Deploy (prod)** — manual approval gate, then deploy

Tooling: GitHub Actions (preferred) or Cloud Build — TBD.

## Acceptance Criteria

- [ ] Pipeline runs on every push to `main` and on every PR
- [ ] Tests must pass before image is built
- [ ] Image is tagged with the git commit SHA
- [ ] Dev deployment happens automatically on merge to `main`
- [ ] Prod deployment requires manual approval
- [ ] Pipeline fails fast on lint errors, test failures, or coverage drop
