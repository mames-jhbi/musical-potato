# STORY-3.2 – /healthz Health Check Endpoint

**Epic:** [EPIC-3 – Cloud Run Deployment](../epics/EPIC-3-cloud-run-deployment.md)
**Status:** 🔵 Backlog | **Points:** 2

## User Story

> As Cloud Run, I want a /healthz endpoint that reflects the true health of the Kafka consumer and Pub/Sub publisher so that unhealthy instances are removed from rotation automatically.

## Details

- FastAPI app runs in a background thread alongside the consumer loop
- `GET /healthz` returns `200 OK` with JSON body when healthy:
  ```json
  { "status": "healthy", "kafka": "connected", "pubsub": "connected" }
  ```
- Returns `503` if Kafka consumer is not polling or Pub/Sub client is unavailable:
  ```json
  { "status": "unhealthy", "kafka": "disconnected", "pubsub": "connected" }
  ```
- Used as Cloud Run liveness probe

## Acceptance Criteria

- [ ] `GET /healthz` returns 200 when both Kafka and Pub/Sub are healthy
- [ ] Returns 503 if Kafka has not successfully polled within last 30 seconds
- [ ] Health state is updated by the consumer loop, not re-checked on each request (avoids blocking)
- [ ] Unit tests cover healthy and unhealthy states
