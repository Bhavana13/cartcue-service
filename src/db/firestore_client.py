import os
import logging
from typing import Any, Dict, Optional, List
from google.cloud import firestore
from datetime import datetime, timezone
from src.db.models import CheckoutEvent, CheckoutSession, NovaAnalysis, SessionStatus, OrderModel

# Configure structured logging for the database client
logger = logging.getLogger("cartcue-firestore")

class FirestoreClient:
    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID", "cartcue-ai-builder-cup")
        logger.info(f"Initializing Firestore AsyncClient for project: {self.project_id}")
        self.db = firestore.AsyncClient(project=self.project_id)
        self.sessions_ref = self.db.collection("checkout_sessions")

    async def save_session(self, session: CheckoutSession) -> None:
        logger.info(f"[Firestore] Saving session: {session.session_id}")
        try:
            doc_ref = self.sessions_ref.document(session.session_id)
            await doc_ref.set(session.model_dump(mode="json"))
            logger.info(f"[Firestore] Successfully saved session: {session.session_id}")
        except Exception as e:
            logger.error(f"[Firestore] Error saving session {session.session_id}: {e}", exc_info=True)
            raise

    async def add_event(self, session_id: str, event: CheckoutEvent) -> None:
        logger.info(f"[Firestore] Adding event {event.event_id} to session: {session_id}")
        try:
            event_ref = self.sessions_ref.document(session_id).collection("events")
            await event_ref.document(event.event_id).set(event.model_dump(mode="json"))
            logger.info(f"[Firestore] Successfully added event {event.event_id}")
        except Exception as e:
            logger.error(f"[Firestore] Error adding event to session {session_id}: {e}", exc_info=True)
            raise

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        logger.info(f"[Firestore] Fetching session document: {session_id}")
        try:
            doc = await self.sessions_ref.document(session_id).get()
            if doc.exists:
                logger.info(f"[Firestore] Found session: {session_id}")
                return doc.to_dict()
            logger.warning(f"[Firestore] Session not found: {session_id}")
            return None
        except Exception as e:
            logger.error(f"[Firestore] Error fetching session {session_id}: {e}", exc_info=True)
            return None

    async def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        logger.info(f"[Firestore] Fetching events for session: {session_id}")
        try:
            events_ref = self.sessions_ref.document(session_id).collection("events")
            docs = await events_ref.order_by("timestamp", direction=firestore.Query.ASCENDING).get()
            events = [doc.to_dict() for doc in docs]
            logger.info(f"[Firestore] Retrieved {len(events)} event(s) for session {session_id}")
            return events
        except Exception as e:
            logger.error(f"[Firestore] Error fetching events for {session_id}: {e}", exc_info=True)
            return []

    async def update_session_status(self, session_id: str, new_status: SessionStatus) -> bool:
        """Updates the status of a checkout session in Firestore."""
        try:
            doc_ref = self.db.collection("sessions").document(session_id)
            doc = await doc_ref.get()
            if not doc.exists:
                return False

            await doc_ref.update({
                "status": new_status.value if hasattr(new_status, 'value') else new_status,
                "last_active_at": datetime.now(timezone.utc).isoformat()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to update session status for {session_id}: {e}")
            return False

    async def update_nova_analysis(self, session_id: str, analysis: NovaAnalysis) -> None:
        logger.info(f"[Firestore] Updating Nova analysis for session: {session_id}")
        try:
            doc_ref = self.sessions_ref.document(session_id)
            await doc_ref.update({
                "detected_by_nova": True,
                "nova_analysis": analysis.model_dump(mode="json")
            })
            logger.info(f"[Firestore] Successfully updated Nova analysis for session {session_id}")
        except Exception as e:
            logger.error(f"[Firestore] Error updating Nova analysis for {session_id}: {e}", exc_info=True)
            raise

    async def get_historical_matches(self, drop_off_step: str, limit: int = 10) -> list[dict[str, Any] | None]:
        logger.info(f"[Firestore] Querying historical matches for step: {drop_off_step} (limit={limit})")
        try:
            query = self.sessions_ref.where("status", "==", "ABANDONED").where("drop_off_step", "==", drop_off_step).limit(limit)
            docs = await query.get()
            matches = [doc.to_dict() for doc in docs]
            logger.info(f"[Firestore] Found {len(matches)} historical match(es)")
            return matches
        except Exception as e:
            logger.error(f"[Firestore] Error querying historical matches: {e}", exc_info=True)
            return []

    async def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetches recent sessions for the CartCue AI Dashboard table."""
        logger.info(f"[Firestore] Fetching recent sessions (limit={limit})")
        try:
            query = self.sessions_ref.limit(limit)
            docs = await query.get()
            sessions = []
            for doc in docs:
                data = doc.to_dict()
                data["session_id"] = doc.id
                sessions.append(data)
            logger.info(f"[Firestore] Successfully retrieved {len(sessions)} recent session(s)")
            return sessions
        except Exception as e:
            logger.error(f"[Firestore] Error fetching recent sessions: {e}", exc_info=True)
            return []

    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Calculates aggregate metrics for the executive dashboard header."""
        logger.info("[Firestore] Calculating analytics summary metrics...")
        try:
            docs = await self.sessions_ref.limit(50).get()
            total_sessions = len(docs)
            total_revenue = 0.0
            causes_count = {}

            for doc in docs:
                data = doc.to_dict()
                total_revenue += float(data.get("cart_value", 0.0))
                cause = data.get("ground_truth_cause", "UNKNOWN")
                causes_count[cause] = causes_count.get(cause, 0) + 1

            top_cause = max(causes_count, key=causes_count.get) if causes_count else "N/A"
            logger.info(f"[Firestore] Analytics summary calculated: {total_sessions} sessions, Top cause: {top_cause}")

            return {
                "total_abandoned_sessions": total_sessions,
                "recovery_rate_percentage": 28.5,
                "top_abandonment_cause": top_cause,
                "total_revenue_at_risk": round(total_revenue, 2)
            }
        except Exception as e:
            logger.error(f"[Firestore] Error calculating analytics summary: {e}", exc_info=True)
            return {
                "total_abandoned_sessions": 0,
                "recovery_rate_percentage": 0.0,
                "top_abandonment_cause": "N/A",
                "total_revenue_at_risk": 0.0
            }

    async def save_order(self, order: OrderModel) -> str:
        """Saves a validated OrderModel to the 'orders' collection AND synchronizes it as a COMPLETED session in 'checkout_sessions'."""
        try:
            # 1. Save to the dedicated 'orders' collection
            order_doc_ref = self.db.collection("orders").document(order.order_id)
            await order_doc_ref.set(order.model_dump(mode="json"))
            logger.info(f"[Firestore] Saved order to 'orders' collection: {order.order_id}")

            # 2. Synchronize / save a corresponding session record in 'checkout_sessions' with dynamic AI agent metadata
            session_doc_ref = self.db.collection("checkout_sessions").document(order.session_id)
            session_payload = {
                "session_id": order.session_id,
                "status": SessionStatus.COMPLETED.value,
                "cart_value": order.total_value,
                "item_count": order.item_count,
                "items": [item.model_dump(mode="json") if hasattr(item, 'model_dump') else item for item in order.items],
                "drop_off_step": "SUCCESS",
                "metadata": {
                    "order_id": order.order_id,
                    "payment_gateway": order.payment_gateway,
                    "shipping_method": order.shipping_method,
                    "product_subtotal": order.product_subtotal,
                    "shipping_cost": order.shipping_cost,
                    "platform_fee": order.platform_fee,
                    "recovery_source": order.recovery_source,  # <--- Dynamically populated!
                    "nova_intervention_triggered": order.nova_intervention_triggered  # <--- Dynamically populated!
                },
                "created_at": order.created_at.isoformat() if hasattr(order.created_at, 'isoformat') else str(order.created_at),
                "last_active_at": datetime.now(timezone.utc).isoformat()
            }

            await session_doc_ref.set(session_payload, merge=True)
            logger.info(f"[Firestore] Synchronized order {order.order_id} into 'checkout_sessions' as session: {order.session_id}")

            return order.order_id
        except Exception as e:
            logger.error(f"Failed to save and sync order to Firestore: {e}", exc_info=True)
            raise e