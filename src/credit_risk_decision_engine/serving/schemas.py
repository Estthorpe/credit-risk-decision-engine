from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ScoreRequest(BaseModel):
    """
    Schema for a credit risk scoring request.
    """
    request_id: Optional[str] = Field(default=None, description="Client-provided request identifier")
    features: Dict[str, Any] = Field(..., description="Raw feature dict (column -> value)")


class ScoreResponse(BaseModel):
    request_id: str
    pd: float
    decision: str
    reason_codes: List[str]
    model_version: str
    input_schema_version: str
    latency_ms: float
    