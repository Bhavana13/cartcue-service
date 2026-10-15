import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import List
from src.db.firestore_client import FirestoreClient
from src.db.models import CauseCategory, CheckoutSession, CheckoutEvent, SessionStatus

# Configure structured logging for the simulator
logger = logging.getLogger("cartcue-simulator")

class SessionSimulator:
    def __init__(self, db_client: FirestoreClient):
        self.db = db_client

    async def generate_scenario(self, cause: CauseCategory) -> str:
        session_id = f"cs_{uuid.uuid4().hex[:8]}"
        now = datetime.utcnow()
        cart_val = round(random.uniform(45.0, 220.0), 2)

        logger.info(f"[Simulator] Generating mock scenario for cause: {cause.value} (ID: {session_id})")

        session = CheckoutSession(
            session_id=session_id,
            created_at=now,
            last_active_at=now + timedelta(minutes=random.randint(1, 5)),  # Fixed field name match
            status=SessionStatus.ABANDONED,
            cart_value=cart_val,
            item_count=random.randint(1, 4),
            ground_truth_cause=cause,
            drop_off_step="SHIPPING_METHOD"
        )

        events: List[CheckoutEvent] = [
            CheckoutEvent(
                event_id=f"evt_{uuid.uuid4().hex[:6]}",
                timestamp=now,
                event_type="STEP_TRANSITION",
                step_name="CART",
                metadata={"cart_value": cart_val}
            )
        ]

        if cause == CauseCategory.SHIPPING_COST_SURPRISE:
            session.drop_off_step = "SHIPPING_METHOD"
            shipping_fee = round(cart_val * random.uniform(0.12, 0.25), 2)
            events.append(
                CheckoutEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:6]}",
                    timestamp=now + timedelta(seconds=20),
                    event_type="STEP_TRANSITION",
                    step_name="SHIPPING_METHOD",
                    metadata={
                        "shipping_cost_shown": shipping_fee,
                        "added_percentage": round((shipping_fee / cart_val) * 100, 1)
                    }
                )
            )

        elif cause == CauseCategory.COUPON_PROMO_FAILURE:
            session.drop_off_step = "REVIEW"
            events.append(
                CheckoutEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:6]}",
                    timestamp=now + timedelta(seconds=35),
                    event_type="PROMO_ATTEMPT",
                    step_name="REVIEW",
                    metadata={
                        "promo_code": "SAVE50",
                        "promo_status": "EXPIRED",
                        "error_message": "Promo code SAVE50 expired on 2026-08-31."
                    }
                )
            )

        elif cause == CauseCategory.PAYMENT_FRICTION:
            session.drop_off_step = "PAYMENT"
            events.append(
                CheckoutEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:6]}",
                    timestamp=now + timedelta(seconds=45),
                    event_type="PAYMENT_ATTEMPT",
                    step_name="PAYMENT",
                    metadata={
                        "gateway_response": "DECLINED",
                        "error_code": "ERR_PAYMENT_GATEWAY_TIMEOUT",
                        "payment_method": "CREDIT_CARD"
                    }
                )
            )

        elif cause == CauseCategory.TECHNICAL_DROP_OFF:
            session.drop_off_step = "SHIPPING_METHOD"
            events.append(
                CheckoutEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:6]}",
                    timestamp=now + timedelta(seconds=15),
                    event_type="STEP_TRANSITION",
                    step_name="SHIPPING_METHOD",
                    metadata={"latency_ms": 4200, "http_status": 504, "error": "Gateway Timeout"}
                )
            )

        try:
            await self.db.save_session(session)
            for event in events:
                await self.db.add_event(session_id, event)
            logger.info(f"[Simulator] Successfully saved scenario {session_id} with {len(events)} event(s) to Firestore.")
        except Exception as e:
            logger.error(f"[Simulator] Error saving scenario {session_id}: {e}", exc_info=True)

        return session_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    client = FirestoreClient()
    sim = SessionSimulator(client)

    async def run_batch():
        logger.info("Starting batch simulation run for all cause categories...")
        for cause in CauseCategory:
            if cause != CauseCategory.UNKNOWN:
                await sim.generate_scenario(cause)
        logger.info("Batch simulation run completed successfully.")

    asyncio.run(run_batch())