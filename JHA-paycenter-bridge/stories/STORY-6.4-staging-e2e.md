# STORY-6.4 – Staging End-to-End Validation

**Epic:** [EPIC-6 – Testing & Quality Assurance](../epics/EPIC-6-testing.md)
**Status:** 🔵 Backlog | **Points:** 3

## User Story

> As a platform engineer, I want a validated end-to-end test in the staging environment using real JH EES connectivity so that we confirm the full pipeline works before production cutover.

## Details

Pre-conditions:
- Staging Cloud Run service deployed
- JH sandbox Kafka credentials provisioned in staging Secret Manager
- JH integration team able to inject synthetic test events per rail type

Test steps:
1. JH team injects a synthetic Zelle Event 800 into the sandbox Kafka topic
2. Confirm message appears in `payment-events` Pub/Sub topic within 10 seconds
3. Confirm message has correct `rail_type = ZELLE` attribute
4. Repeat for RTP and FedNow
5. JH team injects a malformed message — confirm it appears in `payment-events-dead-letter`

## Acceptance Criteria

- [ ] At least one confirmed end-to-end message flow per rail type (Zelle, RTP, FedNow)
- [ ] Messages appear in Pub/Sub within 10 seconds of Kafka injection
- [ ] Message attributes match expected values
- [ ] Malformed message appears in dead letter topic
- [ ] Cloud Monitoring shows corresponding metric increments for each test event

## Notes

- Blocked on JH sandbox environment access (Appendix A Q17)
- Requires coordination with JH integration team to inject test events
