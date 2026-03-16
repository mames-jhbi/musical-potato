# STORY-3.3 – Secret Manager Volume Mounts for Credentials

**Epic:** [EPIC-3 – Cloud Run Deployment](../epics/EPIC-3-cloud-run-deployment.md)
**Status:** 🔵 Backlog | **Points:** 3

## User Story

> As a platform engineer, I want all Kafka credentials and SSL certs loaded from GCP Secret Manager at runtime so that no secrets are stored in the container image or environment variables.

## Details

- Secrets mounted as files via Cloud Run Secret Manager volume mounts (not env vars)
- Mount paths:
  - `/secrets/kafka-username`
  - `/secrets/kafka-password`
  - `/secrets/kafka-ssl-ca-cert` (if mTLS required)
  - `/secrets/kafka-ssl-client-cert` (if mTLS required)
  - `/secrets/kafka-ssl-client-key` (if mTLS required)
- Application reads secrets from mount paths at startup
- Secret names and mount paths configurable via env vars

## Acceptance Criteria

- [ ] Service reads Kafka credentials from Secret Manager mount paths at startup
- [ ] Credentials never appear in `docker inspect` or Cloud Run env var list
- [ ] Missing secret file at startup raises a clear error with the missing path
- [ ] Unit tests mock file reads and verify correct config is assembled
