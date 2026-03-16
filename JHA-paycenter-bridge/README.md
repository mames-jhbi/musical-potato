# JHA PayCenter Payment Events Bridge

> **Python · Kafka → GCP Pub/Sub · Cloud Run · Terraform**

A cloud-native bridge service that consumes real-time payment events from Jack Henry's Enterprise Event System (EES) and publishes them to a unified GCP Pub/Sub topic for downstream processing.

## Payment Rails Supported

| Rail | Network | Operator |
|------|---------|----------|
| **Zelle** | Zelle Network | Early Warning Services |
| **RTP** | Real-Time Payments | The Clearing House (TCH) |
| **FedNow** | FedNow Service | Federal Reserve |

## Architecture

```
JH PayCenter EES  →  Kafka (Event 800)  →  Python Bridge (Cloud Run)  →  GCP Pub/Sub
```

## Repository Structure

```
epics/          # Epic-level descriptions and requirements
stories/        # Individual user stories per epic
```

## Epics

| Epic | Title | Status |
|------|-------|--------|
| [EPIC-1](epics/EPIC-1-kafka-consumer.md) | Kafka Consumer Service | 🔵 Backlog |
| [EPIC-2](epics/EPIC-2-pubsub-integration.md) | GCP Pub/Sub Integration | 🔵 Backlog |
| [EPIC-3](epics/EPIC-3-cloud-run-deployment.md) | Cloud Run Deployment | 🔵 Backlog |
| [EPIC-4](epics/EPIC-4-terraform-iac.md) | Terraform Infrastructure as Code | 🔵 Backlog |
| [EPIC-5](epics/EPIC-5-observability.md) | Observability & Operations | 🔵 Backlog |
| [EPIC-6](epics/EPIC-6-testing.md) | Testing & Quality Assurance | 🔵 Backlog |

## Tech Stack

- **Language:** Python 3.11
- **Kafka Client:** confluent-kafka-python
- **Pub/Sub Client:** google-cloud-pubsub
- **Health API:** FastAPI / Uvicorn
- **Metrics:** OpenTelemetry + GCP exporter
- **Runtime:** Google Cloud Run (min-instances = 1)
- **IaC:** Terraform >= 1.6
- **Secrets:** GCP Secret Manager

## Prerequisites

Before development can begin on Epic 1 and Epic 2, the following must be resolved with the JH PayCenter integration team:

- Kafka broker hostname(s), port, and topic name for Event 800
- Authentication mechanism (SASL or mTLS) and credential provisioning process
- Confirmation of Event 800 schema per rail type (Zelle, RTP, FedNow)
- Network connectivity path (public internet vs VPN/peering) from GCP to JH Kafka

See [Appendix A of the design document](docs/design-document.md) for the full list of integration questions.

---

*Design Document:* `docs/JHA_PayCenter_Bridge_Design_Document.docx`
