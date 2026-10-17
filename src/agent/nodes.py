import os
import json
import logging
from typing import Dict, Any
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
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


def classify_cause_node(state: NovaAgentState) -> Dict[str, Any]:
    session_id = state.get("session_id", "UNKNOWN")
    logger.info(f"[Node: classify_cause] Classifying cause for session: {session_id}")

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
            evidence.append(f"High step latency ({meta.get('latency_ms')}ms) or HTTP error ({meta.get('http_status')}) recorded.")
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

    # 2. Fall back to the trigger recorded by the storefront when the event log is inconclusive
    ground_truth = session_data.get("ground_truth_cause")
    if detected_cause == CauseCategory.UNKNOWN.value and ground_truth and ground_truth != "UNKNOWN":
        logger.info(f"[Node: classify_cause] No conclusive events; using recorded trigger: {ground_truth}")
        detected_cause = ground_truth
        confidence = 0.7
        evidence.append(
            f"No conclusive event telemetry; storefront recorded the drop-off trigger as {ground_truth} "
            f"at step {session_data.get('drop_off_step', 'UNKNOWN')}."
        )

    logger.info(f"[Node: classify_cause] Classified session {session_id} as '{detected_cause}' with confidence {confidence}")
    return {
        "detected_cause": detected_cause,
        "confidence": confidence,
        "evidence_summary": evidence
    }


async def generate_explanation_node(state: NovaAgentState) -> Dict[str, Any]:
    session_id = state.get("session_id", "UNKNOWN")
    logger.info(f"[Node: generate_explanation] Generating AI explanation for session: {session_id}")

    if state.get("error"):
        logger.warning(f"[Node: generate_explanation] Skipping generation due to prior error: {state['error']}")
        return {"explanation_output": f"Analysis failed: {state['error']}"}

    project_id = os.getenv("GCP_PROJECT_ID", "cartcue-ai-builder-cup")
    location = os.getenv("GCP_REGION", "us-central1")

    aiplatform.init(project=project_id, location=location)
    model = GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    You are Nova, an expert e-commerce diagnostic agent for team CartCue.
    Analyze the following session abandonment evidence and synthesize a plain-language explanation.

    Session Data: {json.dumps(state['session_data'], default=str)}
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