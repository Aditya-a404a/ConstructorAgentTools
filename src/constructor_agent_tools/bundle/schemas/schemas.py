from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class BundleProduct(BaseModel):
    id: str
    name: str
    category: str
    price: float
    rating: float = 5.0
    facets: Dict[str, Any] = Field(default_factory=dict)
    score: float = 1.0  # Search relevance or popularity score

class CategoryConstraint(BaseModel):
    slot_id: str
    category: str

class MinRatingConstraint(BaseModel):
    slot_id: Optional[str] = None  # None applies to the whole bundle
    min_rating: float

class PriceConstraint(BaseModel):
    max_total_price: Optional[float] = None
    max_slot_price: Dict[str, float] = Field(default_factory=dict)

class CompatibilityConstraint(BaseModel):
    facet_name: str
    match_type: str = "exact"  # e.g., "exact" means facets must match exactly

class BundleRule(BaseModel):
    rule_name: str
    description: str
    category_constraints: List[CategoryConstraint] = Field(default_factory=list)
    min_rating_constraints: List[MinRatingConstraint] = Field(default_factory=list)
    price_constraints: Optional[PriceConstraint] = None
    compatibility_constraints: List[CompatibilityConstraint] = Field(default_factory=list)
    required_product_ids: List[str] = Field(default_factory=list)
    excluded_product_ids: List[str] = Field(default_factory=list)

class BundleSlot(BaseModel):
    slot_id: str
    display_name: str

class BundleSolutionRequest(BaseModel):
    slots: List[BundleSlot]
    products: List[BundleProduct]
    rules: List[BundleRule]

class BundleSolution(BaseModel):
    products: Dict[str, BundleProduct]  # slot_id -> BundleProduct
    total_price: float
    average_rating: float
    total_score: float

class BundleSolutionResponse(BaseModel):
    solutions: List[BundleSolution]
    total_found: int
