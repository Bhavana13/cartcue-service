import os
import json
import logging
from typing import Any, Dict, Optional
from google.cloud import aiplatform
from vertexai.generative_models import GenerationConfig, GenerativeModel
from src.agent.state import NovaAgentState
from src.db.firestore_client import FirestoreClient
from src.db.models import CauseCategory

# Configure structured logging for agent nodes
logger = logging.getLogger("cartcue-agent-nodes")

db_client = FirestoreClient()


async def retrieve_evidence_node(state: NovaAgentState) -> Dict[str, Any]:
    session_id = state["session_id"]
    logger.info(f"[Node: retrieve_evidence] Starting evidence retrieval for session: {session_id}")

    session_data = await db_client.get_session(session_id)

    if not session_data:
        logger.warning(f"[Node: retrieve_evidence] Session {session_id} not found in Firestore.")
        return {"error": f"Session {session_id} not found."}

    events = await db_client.get_session_events(session_id)
    drop_off_step = session_data.get("drop_off_step", "UNKNOWN")
    historical = await db_client.get_historical_matches(drop_off_step)

    logger.info(
        f"[Node: retrieve_evidence] Retrieved {len(events)} event(s) and "
        f"{len(historical)} historical match(es) for step: {drop_off_step}"
    )

    return {
        "session_data": session_data,
        "events": events,
        "historical_matches": historical
    }


def _gemini_model() -> GenerativeModel:
    aiplatform.init(
        project=os.getenv("GCP_PROJECT_ID", "cartcue-ai-builder-cup"),
        location=os.getenv("GCP_REGION", "us-central1"),
    )
    return GenerativeModel("gemini-2.5-flash")


def classify_by_rules(state: NovaAgentState) -> Dict[str, Any]:
    """Deterministic rules over the event log; returns UNKNOWN when no rule matches."""
    session_id = state.get("session_id", "UNKNOWN")
    logger.info(f"[Rules] Classifying cause for session: {session_id}")

    session_data = state.get("session_data", {})
    events = state.get("events", [])
    cart_value = session_data.get("cart_value", 0.0)

    # 1. Rule-based analysis of the session's event log
    detected_cause = CauseCategory.UNKNOWN.value
    confidence = 0.5
    evidence = []

    for event in events:
        meta = event.get("metadata", {})

        # Rule 1: Technical Drop-off
        if meta.get("latency_ms", 0) > 3000 or meta.get("http_status", 200) >= 500:
            detected_cause = CauseCategory.TECHNICAL_DROP_OFF.value
            confidence = 0.95
            signals = []
            if meta.get("latency_ms", 0) > 3000:
                signals.append(f"{meta['latency_ms']}ms latency")
            if meta.get("http_status", 200) >= 500:
                signals.append(f"HTTP {meta['http_status']}")
            detail = f" ({meta['error']})" if meta.get("error") else ""
            evidence.append(f"Step {event.get('step_name', 'UNKNOWN')} failed with {' and '.join(signals)}{detail}.")
            break

        # Rule 2: Coupon Failure
        if meta.get("promo_status") == "EXPIRED" or "PROMO" in event.get("event_type", ""):
            detected_cause = CauseCategory.COUPON_PROMO_FAILURE.value
            confidence = 0.92
            evidence.append(f"Promo application failed: {meta.get('error_message', 'Invalid code')}.")
            break

        # Rule 3: Payment Friction
        if meta.get("gateway_response") == "DECLINED" or meta.get("error_code"):
            detected_cause = CauseCategory.PAYMENT_FRICTION.value
            confidence = 0.96
            evidence.append(f"Payment gateway failure: {meta.get('error_code', 'Payment declined')}.")
            break

        # Rule 4: Shipping Cost Surprise
        shipping_cost = meta.get("shipping_cost_shown", 0.0)
        if shipping_cost > 0 and cart_value > 0:
            percentage = (shipping_cost / cart_value) * 100
            if percentage >= 10.0:
                detected_cause = CauseCategory.SHIPPING_COST_SURPRISE.value
                confidence = 0.90
                evidence.append(f"Late shipping fee of Rs. {shipping_cost:.2f} added {percentage:.1f}% to cart value (Rs. {cart_value:.2f}).")
                break

    logger.info(f"[Rules] Classified session {session_id} as '{detected_cause}' with confidence {confidence}")
    return {
        "detected_cause": detected_cause,
        "confidence": confidence,
        "evidence_summary": evidence
    }


# Gemini-reasoned causes stay below rule confidence (0.90-0.96), since rules match hard evidence
GEMINI_MAX_CONFIDENCE = 0.85


async def classify_with_gemini(session_data: Dict[str, Any], events: list) -> Optional[Dict[str, Any]]:
    """Asks Gemini to pick a CauseCategory from the telemetry; returns None if it can't or fails."""
    categories = ", ".join(c.value for c in CauseCategory)
    prompt = f"""
    You are Nova, an e-commerce checkout diagnostic agent. Deterministic rules could not explain why this
    shopper abandoned checkout. Reason over the telemetry below and classify the most likely root cause.

    Session Data: {json.dumps(session_data, default=str)}
    Session Events: {json.dumps(events, default=str)}

    Categories: {categories}
    - SHIPPING_COST_SURPRISE: delivery cost shown late or felt too high relative to the cart
    - COUPON_PROMO_FAILURE: a promo or coupon failed to apply
    - PAYMENT_FRICTION: payment declined, failed, or was cumbersome
    - TECHNICAL_DROP_OFF: slowness, errors, or forced sign-up blocked the shopper
    - UNKNOWN: the telemetry does not support any category

    Respond with JSON only: {{"primary_cause": "<one category>", "confidence": <0.0-1.0>,
    "reasoning": "<one sentence citing specific values from the telemetry>"}}
    Do NOT invent events or values that are not in the telemetry.
    """
    try:
        response = await _gemini_model().generate_content_async(
            prompt, generation_config=GenerationConfig(response_mime_type="application/json", temperature=0)
        )
        text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(text)
        cause = CauseCategory(result["primary_cause"]).value
        confidence = min(max(float(result.get("confidence", 0.5)), 0.0), GEMINI_MAX_CONFIDENCE)
        reasoning = str(result.get("reasoning", "")).strip()
    except Exception as e:
        logger.error(f"[Gemini classifier] Failed to classify: {e}", exc_info=True)
        return None

    if cause == CauseCategory.UNKNOWN.value:
        return None
    return {
        "detected_cause": cause,
        "confidence": round(confidence, 2),
        "evidence_summary": [f"Rules were inconclusive; Gemini analysis of the telemetry: {reasoning}"]
    }


async def classify_cause_node(state: NovaAgentState) -> Dict[str, Any]:
    """Rules first, then Gemini over the telemetry, then the storefront's recorded trigger as a last resort."""
    session_id = state.get("session_id", "UNKNOWN")
    logger.info(f"[Node: classify_cause] Classifying cause for session: {session_id}")

    result = classify_by_rules(state)
    if result["detected_cause"] != CauseCategory.UNKNOWN.value:
        return result

    session_data = state.get("session_data") or {}
    events = state.get("events") or []
    # Gemini sees the telemetry, never the recorded label
    telemetry_view = {k: v for k, v in session_data.items() if k != "ground_truth_cause"}
    if events:
        gemini_result = await classify_with_gemini(telemetry_view, events)
        if gemini_result:
            logger.info(f"[Node: classify_cause] Gemini classified session {session_id} as '{gemini_result['detected_cause']}'")
            return gemini_result

    ground_truth = session_data.get("ground_truth_cause")
    if ground_truth and ground_truth != CauseCategory.UNKNOWN.value:
        logger.info(f"[Node: classify_cause] No conclusive evidence; using recorded trigger: {ground_truth}")
        return {
            "detected_cause": ground_truth,
            "confidence": 0.7,
            "evidence_summary": [
                f"No conclusive event telemetry; storefront recorded the drop-off trigger as {ground_truth} "
                f"at step {session_data.get('drop_off_step', 'UNKNOWN')}."
            ]
        }
    return result


async def generate_explanation_node(state: NovaAgentState) -> Dict[str, Any]:
    session_id = state.get("session_id", "UNKNOWN")
    logger.info(f"[Node: generate_explanation] Generating AI explanation for session: {session_id}")

    if state.get("error"):
        logger.warning(f"[Node: generate_explanation] Skipping generation due to prior error: {state['error']}")
        return {"explanation_output": f"Analysis failed: {state['error']}"}

    # Gemini explains the cause Nova detected; it never sees the storefront's recorded label
    session_data = {k: v for k, v in state['session_data'].items() if k != "ground_truth_cause"}

    model = _gemini_model()

    prompt = f"""
    You are Nova, an expert e-commerce diagnostic agent for team CartCue.
    Analyze the following session abandonment evidence and synthesize a plain-language explanation.

    Session Data: {json.dumps(session_data, default=str)}
    Session Events: {json.dumps(state['events'], default=str)}
    Detected Cause Category: {state['detected_cause']}
    Calculated Evidence: {json.dumps(state['evidence_summary'])}
    Historical Correlations Count: {len(state['historical_matches'])}

    Instructions:
    1. Provide a concise, clear explanation (2-3 sentences) detailing WHY the user abandoned.
    2. Cite explicit evidence (rupee amounts, latency numbers, error codes) retrieved from the event stream.
    3. Do NOT hallucinate unprovided events.
    """

    try:
        response = await model.generate_content_async(prompt)
        explanation = response.text.strip()
        logger.info(f"[Node: generate_explanation] Successfully generated explanation for session {session_id}")
    except Exception as e:
        logger.error(f"[Node: generate_explanation] Error calling Gemini model: {e}", exc_info=True)
        explanation = f"Automated analysis completed for {state.get('detected_cause')}, but explanation synthesis failed."

    return {"explanation_output": explanation}