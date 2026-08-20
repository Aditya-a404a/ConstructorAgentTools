from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchProduct(BaseModel):
    """A product returned from a Constructor.io search/browse API call."""
    id: str
    title: str
    category: str = ""
    price: float = 0.0
    relevance_score: float = 1.0  # Search relevance score from Constructor API
    margin: float = 0.0  # Profit margin (business metric)
    inventory: int = 100  # Stock level
    is_sponsored: bool = False  # Whether this is a paid placement
    facets: Dict[str, Any] = Field(default_factory=dict)  # Product attributes for similarity


class MerchandisingParams(BaseModel):
    """
    Weight parameters controlling the QUBO optimization trade-offs.
    
    alpha: Weight for relevance score (higher = favor relevant products).
    beta: Weight for business metrics like margin/inventory (higher = favor profitable products).
    gamma: Weight for diversity penalty (higher = penalize similar products appearing together).
    num_slots: Number of products to select for the final ranked page.
    """
    alpha: float = 1.0   # Relevance weight
    beta: float = 0.5    # Business boost weight
    gamma: float = 0.3   # Diversity penalty weight
    num_slots: int = 10  # Number of products to select


class OptimizeRankingRequest(BaseModel):
    """Request to optimize product ranking using QUBO."""
    products: List[SearchProduct]
    params: MerchandisingParams = Field(default_factory=MerchandisingParams)


class RankedProduct(BaseModel):
    """A product in the final optimized ranking with its component scores."""
    product: SearchProduct
    relevance_contribution: float  # alpha * relevance_score
    business_contribution: float   # beta * business_metric
    diversity_penalty: float       # Total similarity penalty with other selected products
    final_score: float             # Net score contribution


class OptimizeRankingResponse(BaseModel):
    """Response containing the optimized product ranking."""
    ranked_products: List[RankedProduct]
    total_objective_value: float  # The minimized QUBO objective value
    num_candidates: int           # Total products considered
    num_selected: int             # Products selected for final ranking
