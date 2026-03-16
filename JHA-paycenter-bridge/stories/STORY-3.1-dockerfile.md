# STORY-3.1 – Dockerfile and Multi-Stage Build

**Epic:** [EPIC-3 – Cloud Run Deployment](../epics/EPIC-3-cloud-run-deployment.md)
**Status:** 🔵 Backlog | **Points:** 2

## User Story

> As a platform engineer, I want a minimal, multi-stage Docker image so that the container is small, fast to pull, and has a minimal attack surface.

## Details

- Base image: `python:3.11-slim`
- Stage 1 (builder): install Poetry, export `requirements.txt`, install deps into `/install`
- Stage 2 (runtime): copy `/install` and `src/` only — no build tools
- Run as non-root user `appuser`
- No secrets, no `.env` files, no credentials in any layer

## Acceptance Criteria

- [ ] Final image is under 500 MB
- [ ] Container runs as non-root user
- [ ] `docker inspect` shows no environment variables containing credentials
- [ ] Image builds successfully in CI
- [ ] `docker scout cves` shows no critical CVEs in the base image
