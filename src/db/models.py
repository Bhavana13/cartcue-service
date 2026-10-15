from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

class CauseCategory(str, Enum):
    SHIPPING_COST_SURPRISE = "SHIPPING_COST_SURPRISE"
    COUPON_PROMO_FAILURE = "COUPON_PROMO_FAILURE"
    PAYMENT_FRICTION = "PAYMENT_FRICTION"
    TECHNICAL_DROP_OFF = "TECHNICAL_DROP_OFF"
    UNKNOWN = "UNKNOWN"

class SessionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ABANDONED = "ABANDONED"
    COMPLETED = "COMPLETED"

class CheckoutEvent(BaseModel):
    event_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str
    step_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NovaAnalysis(BaseModel):
    primary_cause: CauseCategory
    confidence_score: float = Field(ge=0.0, le=1.0)
    explanation: str
    evidence_summary: List[str] = Field(default_factory=list)
    analysed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class WishlistHesitationRequest(BaseModel):
    item_name: str
    price: float
    user_id: Optional[str] = "guest_user"

class CartItem(BaseModel):
    id: Optional[int] = None
    name: str
    price: float
    category: Optional[str] = "General"

class CheckoutSession(BaseModel):
    session_id: str
    item_id: str = "CartCue"
    items: List[CartItem] = Field(default_factory=list)  # <-- Added items list
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: SessionStatus = SessionStatus.ABANDONED
    cart_value: float
    item_count: int
    drop_off_step: Optional[str] = "CHECKOUT"
    ground_truth_cause: Optional[CauseCategory] = CauseCategory.UNKNOWN
    detected_by_nova: bool = False
    nova_analysis: Optional[NovaAnalysis] = None

    @field_validator('status', mode='before')
    @classmethod
    def parse_status(cls, v):
        if isinstance(v, str):
            return SessionStatus(v.upper())
        return v

    @field_validator('created_at', 'last_active_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

class CheckoutOrderRequest(BaseModel):
    cart_items: List[dict]
    total_value: float
    payment_method: Optional[str] = "PhonePe"
    shipping_method: Optional[str] = "Standard Delivery (Free)"
    recovery_source: Optional[str] = "ORGANIC"
    nova_intervention_triggered: Optional[bool] = False

class OrderModel(BaseModel):
    order_id: str
    session_id: str
    status: SessionStatus = SessionStatus.COMPLETED
    items: List[CartItem] = Field(default_factory=list)
    item_count: int
    product_subtotal: float
    shipping_cost: float
    platform_fee: float
    total_value: float
    payment_gateway: str
    shipping_method: str
    recovery_source: Optional[str] = "ORGANIC"
    nova_intervention_triggered: Optional[bool] = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator('created_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v