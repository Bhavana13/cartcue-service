# cartcue-service

Backend for **CartCue**, built for the Google AI Builder Cup (Retail & Commerce track).

CartCue gives online stores an AI agent, **Nova**, that finds out *why* a shopper abandoned checkout, recovers the sale with an offer that fixes that exact reason, and tracks what was won back. This service hosts the REST API and the Nova agent.

| | |
|---|---|
| Live app | https://cartcue-ai-builder-cup.web.app |
| API | https://cartcue-service-299645583553.us-central1.run.app |
| API docs (Swagger) | https://cartcue-service-299645583553.us-central1.run.app/docs |
| Frontend repo | https://github.com/Bhavana13/cartcue-web |

## How it works

```
Browser ──HTTPS──▶ Firebase Hosting (React store + dashboard)
                        │ REST / JSON
                        ▼
                  Cloud Run: FastAPI ──▶ Nova agent (LangGraph)
                        │                     │
                        ▼                     ▼
                    Firestore        Gemini 2.5 Flash (Vertex AI)
```

### Nova's diagnosis (`POST /analyze`)

A LangGraph pipeline: **retrieve evidence → classify → explain**.

1. **Retrieve evidence:** loads the session, its telemetry events and similar past drop-offs from Firestore.
2. **Classify,** stopping at the first source that answers:
   1. **Rules** over the events, at 90–96% confidence:

      | Signal in the events | Cause |
      |---|---|
      | latency > 3 s or HTTP 5xx | `TECHNICAL_DROP_OFF` |
      | expired or failed promo | `COUPON_PROMO_FAILURE` |
      | declined payment or error code | `PAYMENT_FRICTION` |
      | shipping fee ≥ 10% of the cart | `SHIPPING_COST_SURPRISE` |

   2. **Gemini**, only if no rule matched and events exist. It classifies from the telemetry in JSON mode at temperature 0. Its confidence is capped at **0.85**, so hard evidence always ranks higher. Gemini never sees the store's recorded label.
   3. **Recorded label** (`ground_truth_cause`) at 0.70 as a last resort.
3. **Explain:** Gemini writes a 2–3 sentence explanation that cites the evidence.

The result (cause, confidence, evidence, explanation) is cached on the session.

### Cause-matched offers

Nova only spends margin when the diagnosis says price is the problem. One playbook (`RECOVERY_PLAYBOOK` in `src/api/main.py`) drives both the in-app notification and the recovery email:

| Diagnosed cause | Offer |
|---|---|
| `SHIPPING_COST_SURPRISE` | Free delivery on this order |
| `PAYMENT_FRICTION` | Retry with another payment method |
| `TECHNICAL_DROP_OFF` | Guest checkout with the cart restored |
| `COUPON_PROMO_FAILURE` | The 10% discount the shopper expected |
| `UNKNOWN` (price hesitation) | 10% off every item |

### Nova bell offer (`POST /agent/exit-intent`)

The store calls this when a shopper hesitates (closes the bag without paying, or the mouse heads for the tab bar). Nova runs the **rules** on the live bag to pick the cause and offer. Gemini then writes a short notification describing only that offer, with no invented codes or perks. If the shopper dismisses the offer, the store saves the bag as an abandoned session (`origin: NOVA_OFFER_DISMISSED`) for later diagnosis and recovery. If the shopper orders later, checkout reuses that session.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/sessions?limit=N` | Most recent sessions, newest first |
| POST | `/sessions` | Save a session; optional `events` list is stored in its `events` subcollection |
| POST | `/analyze` | Run (or return cached) Nova diagnosis for `session_id` |
| POST | `/sessions/{id}/recover` | Send a mock recovery email matched to the diagnosed cause |
| POST | `/agent/exit-intent` | Nova bell offer: live diagnosis + Gemini copy |
| POST | `/agent/wishlist-hesitation` | Offer when an item is moved to the wishlist |
| POST | `/orders/checkout` | Place an order; completes `session_id` when given |
| GET | `/analytics/summary` | Abandoned/completed counts, revenue at risk, AI-recovered orders and revenue |
| POST | `/simulate?count=N` | Create simulated abandoned sessions with telemetry |
| DELETE | `/sessions/{id}` | Delete everything for a session: its events, its orders and the session document (also cleans orphaned events). Requires the `X-Admin-Key` header |

Full request and response schemas are at `/docs`.

## Data (Firestore)

- `checkout_sessions/{session_id}`: status (`ABANDONED`, `WAITING_ON_USER_TO_ACT`, `COMPLETED`), cart, drop-off step, recorded cause, `origin`, `nova_analysis`, recovery metadata
- `checkout_sessions/{session_id}/events/{event_id}`: telemetry (`event_type`, `step_name`, `timestamp`, `metadata` such as `shipping_cost_shown`, `gateway_response`, `error_code`, `latency_ms`)
- `orders/{order_id}`: items, subtotal, shipping, platform fee, total, payment method, `recovery_source`

## Project structure

```
src/
  api/main.py              FastAPI app, endpoints, recovery playbook
  agent/graph.py           LangGraph pipeline (retrieve → classify → explain)
  agent/nodes.py           Rules, Gemini classifier, Gemini explanation
  agent/state.py           Agent state
  db/models.py             Pydantic models (sessions, events, orders, requests)
  db/firestore_client.py   Firestore access
  generator/simulator.py   Generates abandoned sessions with telemetry for each cause
verify_firestore.py        Prints orders and completed sessions (debugging)
Dockerfile                 Cloud Run image (non-root)
.github/workflows/deploy.yml
```

## Run locally

Requires Python 3.11+ and a Google Cloud project with Firestore and Vertex AI enabled.

```bash
gcloud auth application-default login      # credentials for Firestore + Vertex AI

uv sync                                     # or: python -m venv .venv && pip install -r requirements.txt
uv run uvicorn src.api.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs. The frontend's `npm run dev` talks to this address by default.

Seed one abandoned session per cause:

```bash
uv run python -m src.generator.simulator
```

To use the Firestore emulator instead of a live database, set `FIRESTORE_EMULATOR_HOST=localhost:8080` before starting.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GCP_PROJECT_ID` | `cartcue-ai-builder-cup` | Project for Firestore and Vertex AI |
| `GCP_REGION` | `us-central1` | Vertex AI region |
| `FIRESTORE_DATABASE` | `(default)` | Firestore database ID |
| `ADMIN_API_KEY` | unset | Unlocks `DELETE /sessions/{id}` (send it as `X-Admin-Key`). When unset, DELETE is always refused |

## Deployment

`.github/workflows/deploy.yml` runs on every push to `main`:
1. checks that the Python sources compile
2. builds the Docker image and pushes it to Artifact Registry
3. deploys it to Cloud Run (`us-central1`), setting `FIRESTORE_DATABASE` from the repository variable `PROD_FIRESTORE_DB`

The service authenticates to Firestore and Vertex AI with its Cloud Run service account; no API keys are stored in the code.

## Tech stack

Python 3.11 · FastAPI · LangGraph · Pydantic · Google Cloud Firestore · Vertex AI (Gemini 2.5 Flash) · Docker · Cloud Run · GitHub Actions
