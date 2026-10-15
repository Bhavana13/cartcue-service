from typing import Any, Dict, List, Optional, TypedDict

class NovaAgentState(TypedDict):
    session_id: str
    session_data: Dict[str, Any]
    events: List[Dict[str, Any]]
    historical_matches: List[Dict[str, Any]]
    detected_cause: Optional[str]
    confidence: float
    explanation_output: str
    evidence_summary: List[str]
    error: Optional[str]