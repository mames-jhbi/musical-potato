# EPIC-3 – Cloud Run Deployment

| Field | Value |
|-------|-------|
| **Epic ID** | EPIC-3 |
| **Status** | 🔵 Backlog |
| **Priority** | P1 |
| **Depends On** | EPIC-1, EPIC-2 |
| **Estimate** | 1 sprint |

## Summary

Containerize the Python bridge service and deploy it to Google Cloud Run with the correct runtime configuration for a persistent Kafka consumer — including always-on CPU, min-instances, health checks, and Secret Manager integration.

## Goal

Run the bridge service as a managed, auto-scaling container on Cloud Run with zero secrets baked into the image, a live health check endpoint, and a CI/CD pipeline for image build and deploy.

## Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-3.1 | The service SHALL be packaged as a Docker container using a Python 3.11 slim base image |
| FR-3.2 | Cloud Run SHALL be configured with `min-instances = 1` to prevent consumer group rebalances due to cold starts |
| FR-3.3 | All secrets (Kafka credentials, SSL certs) SHALL be mounted via GCP Secret Manager volume mounts — never baked into the image |
| FR-3.4 | The service SHALL expose a `/healthz` HTTP endpoint (port 8080) returning HTTP 200 when Kafka consumer and Pub/Sub publisher are both healthy |
| FR-3.5 | Environment-specific configuration (topic names, consumer group ID, log level) SHALL be managed via Cloud Run environment variables |
| FR-3.6 | The container image SHALL be built and pushed to Google Artifact Registry via CI/CD |

## Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-3.1 | Container image size SHALL be under 500 MB using multi-stage builds |
| NFR-3.2 | Cloud Run SHALL be configured with CPU always allocated (not throttled on idle) to maintain persistent Kafka connections |
| NFR-3.3 | Memory limit SHALL be set to 512 MB initially |

## Cloud Run Configuration

| Setting | Value |
|---------|-------|
| `min-instances` | 1 |
| `max-instances` | 3 |
| `cpu` | 1 |
| `memory` | 512Mi |
| `cpu-throttling` | disabled (always-on) |
| `port` | 8080 |
| `timeout` | 3600s |

## Acceptance Criteria

- [ ] Container image builds successfully via CI and is pushed to Artifact Registry
- [ ] Cloud Run service deploys and `/healthz` returns 200 within 30 seconds
- [ ] Restarting the service does not produce duplicate Pub/Sub messages beyond at-least-once guarantee
- [ ] No secrets appear in container environment variables or image layers (`docker inspect` check)
- [ ] `docker scout` or equivalent shows no critical CVEs in the base image

## Stories

| Story | Title |
|-------|-------|
| [STORY-3.1](../stories/STORY-3.1-dockerfile.md) | Dockerfile and multi-stage build |
| [STORY-3.2](../stories/STORY-3.2-healthz-endpoint.md) | `/healthz` health check endpoint |
| [STORY-3.3](../stories/STORY-3.3-secret-manager-mounts.md) | Secret Manager volume mounts for credentials |
| [STORY-3.4](../stories/STORY-3.4-cloud-run-service.md) | Cloud Run service configuration |
| [STORY-3.5](../stories/STORY-3.5-cicd-pipeline.md) | CI/CD pipeline for build, test, and deploy |
