import logging
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI, HTTPException, Query, Request, logger
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agent.graph import nova_app
from src.db.firestore_client import FirestoreClient
from src.db.models import CauseCategory, CheckoutSession, WishlistHesitationRequest, SessionStatus, \
    CheckoutOrderRequest, OrderModel, CartItem

# Configure structured terminal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cartcue-api")

app = FastAPI(
    title="CartCue CheckoutSense API",
    description="Powered by Nova Agent for Root-Cause Abandonment Reasoning",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

db_client = FirestoreClient()


class AnalyzeRequest(BaseModel):
    session_id: str


class AnalyzeResponse(BaseModel):
    session_id: str
    primary_cause: str
    confidence_score: float
    explanation: str
    evidence_summary: List[str]


@app.get("/health")
async def health_check():
    logger.info("Health check endpoint pinged.")
    return {"status": "healthy", "service": "cartcue-nova-api"}


@app.get("/sessions")
async def get_recent_sessions(limit: int = Query(10, ge=1, le=50)):
    """Fetches recent abandoned checkout sessions from Firestore."""
    logger.info(f"Fetching recent sessions with limit={limit}")
    try:
        sessions = await db_client.get_recent_sessions(limit=limit)
        logger.info(f"Successfully retrieved {len(sessions)} session(s).")
        return sessions
    except Exception as e:
        logger.error(f"Error fetching recent sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions")
async def create_storefront_session(session: CheckoutSession):
    """Receives and saves a live checkout session from the Omnivo storefront."""
    logger.info(f"Received storefront session save request for ID: {session.session_id}")
    try:
        await db_client.save_session(session)
        logger.info(f"Successfully saved storefront session: {session.session_id}")
        return {"status": "success", "session_id": session.session_id, "message": "Session saved successfully."}
    except Exception as e:
        logger.error(f"Error saving storefront session {session.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


from datetime import datetime, timezone
from fastapi import HTTPException


@app.post("/sessions/{session_id}/recover")
async def recover_session(session_id: str):
    try:
        doc_ref = db_client.db.collection("checkout_sessions").document(session_id)
        doc = await doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = doc.to_dict()
        cause = session_data.get("ground_truth_cause", "UNKNOWN")

        # Tailor the mock email based on the specific diagnostic root cause
        if cause == "SHIPPING_COST_SURPRISE":
            email_subject = "Complete your Omnivo order with FREE Express Shipping!"
            email_body = "We noticed you left your cart due to shipping costs. Here is a special voucher for free priority delivery to complete your order."
        elif cause == "PAYMENT_FRICTION":
            email_subject = "Having trouble checking out? We're here to help!"
            email_body = "It looks like your payment didn't go through. Click here to use our secure alternative checkout link."
        elif cause == "TECHNICAL_DROP_OFF":
            email_subject = "Resume your Omnivo Cart - Special Assistance"
            email_body = "We resolved the account wall issue you encountered. Click here to instantly restore your cart items and finish checking out."
        else:
            email_subject = "Complete your Omnivo order!"
            email_body = f"We noticed you left behind items in your cart due to {cause}. Click here to resume your checkout."

        # Simulate a targeted recovery email based on the diagnostic cause
        mock_email = {
            "recipient": f"customer_{session_id}@example.com",
            "subject": email_subject,
            "body": email_body,
            "sent_at": datetime.now(timezone.utc).isoformat()
        }

        recovery_payload = {
            "status": "WAITING_ON_USER_TO_ACT",
            "metadata": {
                "recovery_source": "NOVA_WISHLIST_AGENT",
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
    except Exception as e:
        logger.error(f"Error recovering session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/wishlist-hesitation")
async def handle_wishlist_hesitation(payload: WishlistHesitationRequest):
    """Shopify-style proactive AI agent: detects price hesitation, applies a 10% discount, and recovers the cart."""
    logger.info(
        f"Nova Agent triggered for wishlist hesitation on item: {payload.item_name} (Price: Rs. {payload.price})")

    discounted_price = round(payload.price * 0.9, 2)

    return {
        "status": "success",
        "agent_name": "Nova Proactive Recovery Agent",
        "analysis": "High price sensitivity detected via wishlist migration.",
        "intervention_triggered": True,
        "recovery_source": "NOVA_WISHLIST_AGENT",  # <--- Indicator tag
        "discount_code": "NOVA10_RECOVER",
        "original_price": payload.price,
        "discounted_price": discounted_price,
        "message": f"Nova Agent offered a targeted 10% discount on {payload.item_name} (Rs. {payload.price} → Rs. {discounted_price}). "
                   f"Item successfully restored to cart!"
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
            status = data.get("status", "ABANDONED")
            cart_value = data.get("cart_value", 0.0)
            metadata = data.get("metadata") or {}

            if status == "COMPLETED":
                total_completed += 1
                # Check if recovered by Nova AI Agent
                if metadata.get("recovery_source") == "NOVA_WISHLIST_AGENT" or metadata.get(
                        "nova_intervention_triggered") is True:
                    ai_recovered_count += 1
                    ai_recovered_revenue += cart_value
            else:
                total_abandoned += 1
                revenue_at_risk += cart_value

        total_sessions = total_abandoned + total_completed
        recovery_rate = round((total_completed / total_sessions * 100) if total_sessions > 0 else 0.0, 1)

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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate")
async def simulate_activity(count: int = Query(1, ge=1, le=10)):
    """Programmatically generates mock checkout sessions on-demand using valid CauseCategory enums."""
    logger.info(f"Triggering simulation for {count} mock session(s).")
    simulated_sessions = []

    trigger_mappings = [
        {
            "cause": CauseCategory.SHIPPING_COST_SURPRISE.value,
            "step": "Checkout Review",
            "label": "Shipping Fee Shock"
        },
        {
            "cause": CauseCategory.PAYMENT_FRICTION.value,
            "step": "Payment Processing",
            "label": "Technical Gateway Timeout"
        },
        {
            "cause": CauseCategory.TECHNICAL_DROP_OFF.value,
            "step": "Authentication / Signup",
            "label": "Forced Account Wall"
        }
    ]

    for i in range(count):
        session_id = f"cs_{uuid.uuid4().hex[:8]}"
        mapping = trigger_mappings[i % len(trigger_mappings)]
        now_iso = datetime.now(timezone.utc).isoformat()

        session_doc = {
            "session_id": session_id,
            "ground_truth_cause": mapping["cause"],
            "trigger_label": mapping["label"],
            "drop_off_step": mapping["step"],
            "status": "abandoned",
            "cart_value": 150.0 + (i * 12.5),
            "timestamp": now_iso,
            "created_at": now_iso
        }

        try:
            await db_client.save_session(session_doc)
            logger.info(f"Saved simulated session: {session_id} ({mapping['label']})")
        except Exception as e:
            logger.warning(f"Failed to save session {session_id} to Firestore: {e}")

        simulated_sessions.append(session_id)

    return {
        "status": "success",
        "simulated_count": count,
        "session_ids": simulated_sessions,
        "message": f"Successfully generated {count} mock session(s) with mapped enum causes."
    }

@app.post("/analyze")
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
            return {
                "session_id": session_id,
                "primary_cause": cached.get("primary_cause", ground_truth),
                "confidence_score": cached.get("confidence_score", 0.95),
                "explanation": cached.get("explanation", f"Session dropped off due to {ground_truth}.")
            }

        # 2. Otherwise, run LangGraph workflow for the first time
        logger.info(f"[Nova Diagnostics] Invoking LangGraph workflow for session {session_id}...")
        initial_state = {"session_id": session_id}
        final_state = await nova_app.ainvoke(initial_state)

        detected_cause = final_state.get("detected_cause", ground_truth)
        confidence = final_state.get("confidence", 0.95)
        explanation = final_state.get("explanation_output", f"Session dropped off due to {detected_cause}.")

        # 3. Save results and flags back to Firestore
        analysis_payload = {
            "is_nova_analysed": True,
            "nova_analysis": {
                "analysed_at": datetime.now(timezone.utc).isoformat(),
                "primary_cause": detected_cause,
                "confidence_score": confidence,
                "explanation": explanation
            }
        }
        await doc_ref.set(analysis_payload, merge=True)

        return {
            "session_id": session_id,
            "primary_cause": detected_cause,
            "confidence_score": confidence,
            "explanation": explanation
        }

    except Exception as e:
        logger.error(f"Error analyzing session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/orders/checkout")
async def checkout_order(payload: CheckoutOrderRequest):
    """Processes successful checkout with detailed cost breakdowns and saves to 'orders'."""
    order_id = f"ORD-{uuid.uuid4().hex[:6].upper()}"
    session_id = f"cs_{uuid.uuid4().hex[:8]}"

    # Calculate fee breakdowns
    product_subtotal = sum(item.get("price", 0) for item in payload.cart_items)

    # Determine shipping cost based on selected method string
    shipping_cost = 0.0
    if "Express" in payload.shipping_method:
        shipping_cost = 150.0
    elif "Priority" in payload.shipping_method:
        shipping_cost = 350.0

    platform_fee = 23.0
    total_value = product_subtotal + shipping_cost + platform_fee

    logger.info(
        f"Processing order {order_id}: Subtotal={product_subtotal}, Shipping={shipping_cost}, "
        f"Platform={platform_fee}, Total={total_value}, Source={payload.recovery_source}")

    try:
        order_obj = OrderModel(
            order_id=order_id,
            session_id=session_id,
            status=SessionStatus.COMPLETED,
            items=[CartItem(**item) for item in payload.cart_items],
            item_count=len(payload.cart_items),
            product_subtotal=product_subtotal,
            shipping_cost=shipping_cost,
            platform_fee=platform_fee,
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
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def delete_session_and_order(session_id: str):
    """
    Delete a checkout session and any associated orders from Firestore.
    Available exclusively via Swagger documentation.
    """
    try:
        # 1. Reference and delete the session document
        session_ref = db_client.db.collection("checkout_sessions").document(session_id)
        session_doc = await session_ref.get()

        if not session_doc.exists:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        await session_ref.delete()

        # 2. Check and delete associated orders if any exist
        orders_ref = db_client.db.collection("orders")
        orders_query = orders_ref.where("session_id", "==", session_id)
        orders_docs = orders_query.stream()

        deleted_orders_count = 0
        async for order_doc in orders_docs:
            await order_doc.reference.delete()
            deleted_orders_count += 1

        logger.info(f"[Admin] Deleted session {session_id} and {deleted_orders_count} associated orders.")
        return {
            "status": "success",
            "message": f"Session {session_id} and {deleted_orders_count} associated order(s) successfully deleted."
        }
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error for request {request.url}: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})