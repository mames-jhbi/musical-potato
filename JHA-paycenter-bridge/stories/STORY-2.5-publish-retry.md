# STORY-2.5 – Publish Retry Logic with Exponential Backoff

**Epic:** [EPIC-2 – GCP Pub/Sub Integration](../epics/EPIC-2-pubsub-integration.md)
**Status:** 🔵 Backlog | **Points:** 3

## User Story

> As a platform engineer, I want transient Pub/Sub publish failures to be retried automatically so that brief GCP outages do not result in message loss or dead lettering.

## Details

- Max retries: 5
- Backoff: exponential — 0.5s → 1s → 2s → 4s → 8s
- Retry on: `google.api_core.exceptions.ServiceUnavailable`, `DeadlineExceeded`, `InternalServerError`
- Do NOT retry on: `PermissionDenied`, `NotFound` (these go straight to DLT)
- After max retries exhausted: route to dead letter with `error_type = PUBLISH_FAILURE`

## Acceptance Criteria

- [ ] Transient errors trigger up to 5 retries with exponential backoff
- [ ] Non-retryable errors skip retries and go straight to DLT
- [ ] Retry count and final outcome are logged in structured format
- [ ] After 5 failed retries, message is routed to DLT
- [ ] Unit tests mock each retry scenario and verify call count and backoff
