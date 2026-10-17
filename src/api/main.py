import logging
import os
import secrets
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import vertexai
from vertexai.generative_models import GenerativeModel

from src.agent.graph import nova_app
from src.agent.nodes import classify_by_rules
from src.db.firestore_client import FirestoreClient
from src.db.models import CauseCategory, CheckoutSession, WishlistHesitationRequest, SessionStatus, \
    CheckoutOrderRequest, ExitIntentRequest, OrderModel, CartItem
from src.generator.simulator import SessionSimulator

# Configure structured terminal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cartcue-api")

vertexai.init(
    project=os.getenv("GCP_PROJECT_ID", "cartcue-ai-builder-cup"),
    location=os.getenv("GCP_REGION", "us-central1"),
)
model = GenerativeModel("gemini-2.5-flash")

app = FastAPI(
    title="CartCue CheckoutSense API",
    description="Powered by Nova Agent for Root-Cause Abandonment Reasoning",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Public demo API; no cookies or credentials are used
    allow_methods=["*"],
    allow_headers=["*"],
)

db_client = FirestoreClient()
simulator = SessionSimulator(db_client)

# Shipping fees keyed by a substring of the storefront's shipping label
SHIPPING_FEES = {"Express": 150.0, "Priority": 350.0}
PLATFORM_FEE = 23.0

INTERNAL_ERROR = "Internal server error. Check service logs for details."

# Nova's recovery playbook: margin is only spent when the diagnosis says price is the problem.
RECOVERY_PLAYBOOK = {
    CauseCategory.SHIPPING_COST_SURPRISE.value: {
        "offer_type": "FREE_SHIPPING",
        "offer_label": "Free delivery on this order",
        "email_subject": "Your Omnivo order now ships free",
        "email_body": "The delivery fee surprised you at checkout, so we've removed it. Your cart is saved: click here to finish with free delivery.",
    },
    CauseCategory.PAYMENT_FRICTION.value: {
        "offer_type": "ALT_PAYMENT",
        "offer_label": "Retry with another payment method",
        "email_subject": "Your payment didn't go through. Your cart is saved",
        "email_body": "The payment gateway timed out, not you. Click here to finish with another payment method (UPI, card or wallet). No changes to your order.",
    },
    CauseCategory.TECHNICAL_DROP_OFF.value: {
        "offer_type": "RESTORE_CART",
        "offer_label": "Guest checkout with your cart restored",
        "email_subject": "Sorry about that. Your Omnivo cart is ready",
        "email_body": "Checkout was slow and asked you to sign up. Click here to finish as a guest with your cart already restored.",
    },
    CauseCategory.COUPON_PROMO_FAILURE.value: {
        "offer_type": "PERCENT_OFF_10",
        "offer_label": "10% off every item in the bag",
        "email_subject": "Your Omnivo discount is waiting in your cart",
        "email_body": "Your promo code didn't apply, so we've added a 10% discount to your saved cart. Click here to finish at the price you expected.",
    },
    CauseCategory.UNKNOWN.value: {
        "offer_type": "PERCENT_OFF_10",
        "offer_label": "10% off every item in the bag",
        "email_subject": "Still thinking it over? Here's 10% off",
        "email_body": "Your items are still in your cart, and we've added 10% off to help you decide. Click here to resume your checkout.",
    },
}


def playbook_for(cause: Optional[str]) -> dict:
    return RECOVERY_PLAYBOOK.get(cause or "", RECOVERY_PLAYBOOK[CauseCategory.UNKNOWN.value])


class AnalyzeResponse(BaseModel):
    session_id: str
    primary_cause: str
    confidence_score: float
    explanation: str
    evidence_summary: List[str]


def require_admin(admin_key: Optional[str]) -> None:
    """Guards destructive endpoints with the ADMIN_API_KEY env var; they stay locked when it is unset."""
    expected = os.getenv("ADMIN_API_KEY")
    if not expected or not admin_key or not secrets.compare_digest(admin_key, expected):
        raise HTTPException(status_code=403, detail="Admin access required")


@app.get("/health")
async def health_check():
    logger.info("Health check endpoint pinged.")
    return {"status": "healthy", "service": "cartcue-nova-api"}


@app.get("/sessions")
async def get_recent_sessions(limit: int = Query(10, ge=1, le=50)):
    """Fetches the most recent checkout sessions from Firestore."""
    logger.info(f"Fetching recent sessions with limit={limit}")
    try:
        sessions = await db_client.get_recent_sessions(limit=limit)
        logger.info(f"Successfully retrieved {len(sessions)} session(s).")
        return sessions
    except Exception as e:
        logger.error(f"Error fetching recent sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR)


@app.post("/sessions")
async def create_storefront_session(session: CheckoutSession):
    """Receives and saves a live checkout session, plus any event telemetry, from the Omnivo storefront."""
    logger.info(f"Received storefront session save request for ID: {session.session_id}")
    try:
        await db_client.save_session(session)
        for event in session.events:
            await db_client.add_event(session.session_id, event)
        logger.info(f"Successfully saved storefront session: {session.session_id} with {len(session.events)} event(s)")
        return {"status": "success", "session_id": session.session_id, "message": "Session saved successfully."}
    except Exception as e:
        logger.error(f"Error saving storefront session {session.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR)


@app.post("/sessions/{session_id}/recover")
async def recover_session(session_id: str):
    try:
        doc_ref = db_client.db.collection("checkout_sessions").document(session_id)
        doc = await doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = doc.to_dict()
        # Prefer Nova's diagnosis over the recorded trigger when one exists
        cause = (session_data.get("nova_analysis") or {}).get("primary_cause") \
            or session_data.get("ground_truth_cause", "UNKNOWN")

        # Tailor the mock email and offer to the diagnosed root cause
        play = playbook_for(cause)
        mock_email = {
            "recipient": f"customer_{session_id}@example.com",
            "subject": play["email_subject"],
            "body": play["email_body"],
            "diagnosed_cause": cause,
            "offer_type": play["offer_type"],
            "offer_label": play["offer_label"],
            "sent_at": datetime.now(timezone.utc).isoformat()
        }

        recovery_payload = {
            "status": SessionStatus.WAITING_ON_USER_TO_ACT.value,
            "metadata": {
                "recovery_source": "NOVA_RECOVERY_EMAIL",
                "nova_intervention_triggered": True,
                "recovery_email": mock_email
            }
        }
        await doc_ref.set(recovery_payload, merge=True)

        return {
            "status": "success",
            "session_id": session_id,
            "message": "Recovery email dispatched successfully",
            "mock_email": mock_email
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recovering session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR)


@app.post("/agent/wishlist-hesitation")
async def handle_wishlist_hesitation(payload: WishlistHesitationRequest):
    """Advanced AI agent: detects user behavior and dynamically offers either a price discount or free express delivery."""
    logger.info(f"Nova Agent triggered for wishlist hesitation on item: {payload.item_name} (Price: Rs. {payload.price})")

    # Simulate AI behavior profiling: Tier-1 / Convenience vs Price-sensitive
    # In a full production app, this would query Gemini based on user history.
    is_convenience_focused = payload.price > 1500  # Example heuristic: higher-ticket items prefer delivery speed/perks

    if is_convenience_focused:
        intervention_type = "EXPRESS_DELIVERY_UPGRADE"
        message = f"Nova Agent analyzed your profile: Since you value speed, we've upgraded your delivery to **Same-Day Priority (Free)** on {payload.item_name}!"
        discount_code = "NOVA_EXPRESS_FREE"
        discounted_price = payload.price
    else:
        intervention_type = "PRICE_DISCOUNT"
        discounted_price = round(payload.price * 0.9, 2)
        message = f"Nova Agent detected price sensitivity: Offered a targeted **10% discount** on {payload.item_name} (Rs. {payload.price} → Rs. {discounted_price})."
        discount_code = "NOVA10_RECOVER"

    return {
        "status": "success",
        "agent_name": "Nova Proactive Multi-Objective Recovery Agent",
        "analysis": f"Behavioral intent classified as {intervention_type}.",
        "intervention_triggered": True,
        "recovery_source": "NOVA_WISHLIST_AGENT",
        "intervention_type": intervention_type,
        "discount_code": discount_code,
        "original_price": payload.price,
        "discounted_price": discounted_price,
        "message": message
    }


@app.get("/analytics/summary")
async def get_analytics_summary():
    """Returns executive metrics including total abandoned sessions, recovery rate, revenue at risk, and AI agent performance stats."""
    try:
        # Fetch all sessions from Firestore
        sessions_ref = db_client.db.collection("checkout_sessions")
        docs = await sessions_ref.get()

        total_abandoned = 0
        total_completed = 0
        revenue_at_risk = 0.0
        ai_recovered_count = 0
        ai_recovered_revenue = 0.0

        for doc in docs:
            data = doc.to_dict()
            status = str(data.get("status", "ABANDONED")).upper()
            cart_value = data.get("cart_value", 0.0)
            metadata = data.get("metadata") or {}

            if status == SessionStatus.COMPLETED.value:
                total_completed += 1
                # Only orders where a Nova intervention was applied count as AI recoveries
                if metadata.get("nova_intervention_triggered") is True:
                    ai_recovered_count += 1
                    ai_recovered_revenue += cart_value
            else:
                total_abandoned += 1
                revenue_at_risk += cart_value

        # Share of at-risk checkouts that Nova won back
        at_risk_sessions = total_abandoned + ai_recovered_count
        recovery_rate = round((ai_recovered_count / at_risk_sessions * 100) if at_risk_sessions > 0 else 0.0, 1)

        return {
            "total_abandoned_sessions": total_abandoned,
            "total_completed_sessions": total_completed,
            "recovery_rate_percentage": recovery_rate,
            "total_revenue_at_risk": round(revenue_at_risk, 2),
            "ai_agent_stats": {
                "recovered_orders_count": ai_recovered_count,
                "recovered_revenue": round(ai_recovered_revenue, 2)
            }
        }
    except Exception as e:
        logger.error(f"Error generating analytics summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR)


@app.post("/simulate")
async def simulate_activity(count: int = Query(1, ge=1, le=10)):
    """Generates mock abandoned sessions, with event telemetry, cycling through every CauseCategory."""
    logger.info(f"Triggering simulation for {count} mock session(s).")
    causes = [c for c in CauseCategory if c != CauseCategory.UNKNOWN]

    simulated_sessions = []
    try:
        for i in range(count):
            simulated_sessions.append(await simulator.generate_scenario(causes[i % len(causes)]))
    except Exception as e:
        logger.error(f"Simulation failed after {len(simulated_sessions)} session(s): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR)

    return {
        "status": "success",
        "simulated_count": len(simulated_sessions),
        "session_ids": simulated_sessions,
        "message": f"Successfully generated {len(simulated_sessions)} mock session(s) with event telemetry."
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_session(payload: dict):
    session_id = payload.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        doc_ref = db_client.db.collection("checkout_sessions").document(session_id)
        doc = await doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        session_data = doc.to_dict()
        ground_truth = session_data.get("ground_truth_cause", "UNKNOWN")

        # 1. CACHE CHECK: Return cached analysis instantly if already analyzed
        if session_data.get("is_nova_analysed") and session_data.get("nova_analysis"):
            logger.info(f"[Nova Diagnostics] Returning cached analysis for session {session_id}")
            cached = session_data["nova_analysis"]
            return AnalyzeResponse(
                session_id=session_id,
                primary_cause=cached.get("primary_cause", ground_truth),
                confidence_score=cached.get("confidence_score", 0.5),
                explanation=cached.get("explanation", f"Session dropped off due to {ground_truth}."),
                evidence_summary=cached.get("evidence_summary", [])
            )

        # 2. Otherwise, run LangGraph workflow for the first time
        logger.info(f"[Nova Diagnostics] Invoking LangGraph workflow for session {session_id}...")
        final_state = await nova_app.ainvoke({"session_id": session_id})

        detected_cause = final_state.get("detected_cause") or CauseCategory.UNKNOWN.value
        confidence = final_state.get("confidence", 0.5)
        explanation = final_state.get("explanation_output", f"Session dropped off due to {detected_cause}.")
        evidence = final_state.get("evidence_summary", [])

        # 3. Save results and flags back to Firestore
        analysis_payload = {
            "is_nova_analysed": True,
            "nova_analysis": {
                "analysed_at": datetime.now(timezone.utc).isoformat(),
                "primary_cause": detected_cause,
                "confidence_score": confidence,
                "explanation": explanation,
                "evidence_summary": evidence
            }
        }
        await doc_ref.set(analysis_payload, merge=True)

        return AnalyzeResponse(
            session_id=session_id,
            primary_cause=detected_cause,
            confidence_score=confidence,
            explanation=explanation,
            evidence_summary=evidence
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR)


@app.post("/agent/exit-intent")
async def handle_exit_intent(payload: ExitIntentRequest):
    """Powers the Nova bell notification: diagnoses the live bag with Nova's rules, picks the matching offer,
    and has Gemini describe only that offer."""
    item_count = len(payload.cart_items)
    item_names = ", ".join(str(item.get("name", "item")) for item in payload.cart_items) or "items"
    subtotal = sum(float(item.get("price", 0)) for item in payload.cart_items)

    # What the shopper is looking at right now, as telemetry for Nova's classifier
    live_state = {
        "session_id": "live-exit-intent",
        "session_data": {"cart_value": subtotal + payload.shipping_fee + PLATFORM_FEE},
        "events": [{"event_type": "STEP_TRANSITION", "step_name": "REVIEW",
                    "metadata": {"shipping_cost_shown": payload.shipping_fee}}],
    }
    diagnosis = classify_by_rules(live_state)
    cause = diagnosis["detected_cause"]
    play = playbook_for(cause)

    if cause == CauseCategory.SHIPPING_COST_SURPRISE.value:
        reason = f"the Rs. {payload.shipping_fee:.0f} delivery fee ({payload.shipping_method or 'selected shipping'})"
    else:
        reason = "no specific friction, so likely price hesitation"

    prompt = (
        f"A retail shopper on Omnivo is hesitating with {item_count} items in their bag: {item_names}. "
        f"Bag subtotal: Rs. {subtotal:.0f}. Nova diagnosed the likely reason as {reason}. "
        f"Act as Nova, an AI retail recovery agent. Write a short, warm in-app notification "
        f"(under 2 sentences) presenting exactly this offer: {play['offer_label']}. The shopper applies it with a button, "
        f"so do not mention coupon codes, and do not invent any other discount, percentage, deadline or perk."
    )

    try:
        response = await model.generate_content_async(prompt)
        ai_message = response.text.strip()
    except Exception as e:
        logger.error(f"Exit-intent Gemini call failed: {e}", exc_info=True)
        ai_message = f"Before you go: Nova has unlocked {play['offer_label'].lower()} just for you."

    return {
        "status": "success",
        "agent_name": "Nova Exit-Intent Agent",
        "message": ai_message,
        "diagnosed_cause": cause,
        "evidence_summary": diagnosis["evidence_summary"],
        "offer_type": play["offer_type"],
        "offer_label": play["offer_label"]
    }


@app.post("/orders/checkout")
async def checkout_order(payload: CheckoutOrderRequest):
    """Processes successful checkout with detailed cost breakdowns and saves to 'orders'."""
    order_id = f"ORD-{uuid.uuid4().hex[:6].upper()}"
    # Completing a captured abandoned session keeps its events and diagnosis on the same document
    session_id = payload.session_id or f"cs_{uuid.uuid4().hex[:8]}"

    # Calculate fee breakdowns
    product_subtotal = sum(item.get("price", 0) for item in payload.cart_items)

    # Determine shipping cost from the selected method, unless a Nova offer waived it
    shipping_cost = 0.0
    if not payload.shipping_fee_waived:
        shipping_cost = next((fee for key, fee in SHIPPING_FEES.items() if key in payload.shipping_method), 0.0)

    total_value = product_subtotal + shipping_cost + PLATFORM_FEE

    logger.info(
        f"Processing order {order_id}: Subtotal={product_subtotal}, Shipping={shipping_cost}, "
        f"Platform={PLATFORM_FEE}, Total={total_value}, Source={payload.recovery_source}")

    try:
        order_obj = OrderModel(
            order_id=order_id,
            session_id=session_id,
            status=SessionStatus.COMPLETED,
            items=[CartItem(**item) for item in payload.cart_items],
            item_count=len(payload.cart_items),
            product_subtotal=product_subtotal,
            shipping_cost=shipping_cost,
            platform_fee=PLATFORM_FEE,
            total_value=total_value,
            payment_gateway=payload.payment_method,
            shipping_method=payload.shipping_method,
            recovery_source=payload.recovery_source,
            nova_intervention_triggered=payload.nova_intervention_triggered,
            created_at=datetime.now(timezone.utc)
        )

        await db_client.save_order(order_obj)
        logger.info(f"Order {order_id} successfully persisted to Firestore 'orders' collection.")

        return {
            "status": "success",
            "order_id": order_id,
            "session_id": session_id,
            "total_value": total_value,
            "message": f"Order {order_id} successfully placed via {payload.payment_method}!"
        }
    except Exception as e:
        logger.error(f"Error processing order {order_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR)


@app.delete("/sessions/{session_id}")
async def delete_session_and_order(session_id: str, x_admin_key: Optional[str] = Header(None)):
    """
    Delete a checkout session and any associated orders from Firestore.
    Requires the X-Admin-Key header to match the ADMIN_API_KEY env var.
    """
    require_admin(x_admin_key)
    try:
        # 1. Reference and delete the session document
        session_ref = db_client.db.collection("checkout_sessions").document(session_id)
        session_doc = await session_ref.get()

        if not session_doc.exists:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        await session_ref.delete()

        # 2. Check and delete associated orders if any exist
        orders_query = db_client.db.collection("orders").where("session_id", "==", session_id)

        deleted_orders_count = 0
        async for order_doc in orders_query.stream():
            await order_doc.reference.delete()
            deleted_orders_count += 1

        logger.info(f"[Admin] Deleted session {session_id} and {deleted_orders_count} associated orders.")
        return {
            "status": "success",
            "message": f"Session {session_id} and {deleted_orders_count} associated order(s) successfully deleted."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=INTERNAL_ERROR)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error for request {request.url}: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})
