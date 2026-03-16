# STORY-2.3 – Ordering Key Configuration by transactionId

**Epic:** [EPIC-2 – GCP Pub/Sub Integration](../epics/EPIC-2-pubsub-integration.md)
**Status:** 🔵 Backlog | **Points:** 2

## User Story

> As a downstream consumer, I want Pub/Sub messages for the same transaction to be delivered in order so that multi-event payment lifecycles (PENDING → PROCESSING → COMPLETED) can be processed correctly.

## Details

- Set Pub/Sub `ordering_key` to `transactionId` from the Event 800 payload
- If `transactionId` is absent or null, publish without ordering key (log a warning)
- Ordering keys require `enable_message_ordering=True` on the publisher client

## Acceptance Criteria

- [ ] Publisher client has `enable_message_ordering=True`
- [ ] `ordering_key` on every published message equals `transactionId` from the payload
- [ ] A message with no `transactionId` is published without an ordering key and a warning is logged
- [ ] Unit tests verify ordering key is set correctly
