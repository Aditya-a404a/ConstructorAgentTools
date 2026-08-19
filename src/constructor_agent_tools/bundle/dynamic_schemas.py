from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CatalogProduct(BaseModel):
    """A product in the catalog with its description and optional pre-computed embedding."""
    id: str
    title: str
    description: str = ""
    category: str = ""
    price: float = 0.0
    rating: float = 5.0
    image_url: Optional[str] = None
    facets: Dict[str, Any] = Field(default_factory=dict)


class SlotRule(BaseModel):
    """A bundle slot with a verbose, LLM-generated compatibility rule description."""
    slot_id: str
    display_name: str
    rule_description: str  # Verbose natural language rule for vector search
    category_hint: Optional[str] = None  # Optional category filter to narrow search


class IntentResult(BaseModel):
    """Output of LLM Call 1: intent classification and slot rules."""
    intent_type: str = "soft"  # "soft" (stylistic) or "hard" (functional)
    slots: List[SlotRule]
    max_total_price: Optional[float] = None


class ScoredCandidate(BaseModel):
    """A candidate product retrieved by vector search, with its similarity score."""
    product: CatalogProduct
    similarity_score: float


class DynamicBundleSolution(BaseModel):
    """A solved bundle: one product per slot, with scores."""
    slot_products: Dict[str, CatalogProduct]  # slot_id -> chosen product
    slot_scores: Dict[str, float]  # slot_id -> similarity score
    total_price: float
    aggregate_similarity: float  # Sum or mean of per-slot similarity scores
