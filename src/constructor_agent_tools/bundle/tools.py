from typing import Any, Dict, List, Optional
from constructor_agent_tools.bundle.schemas import (
    BundleProduct,
    BundleRule,
    BundleSlot,
    BundleSolutionResponse,
)
from constructor_agent_tools.bundle.solver import BundleSolver

def find_best_bundles(
    slots: List[Dict[str, Any]],
    products: List[Dict[str, Any]],
    rules: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Finds the optimal set of product bundles that satisfy all custom merchandising 
    and product-compatibility constraints.

    Args:
        slots: A list of slot dictionaries. Each slot should have 'slot_id' and 'display_name'.
            Example: [{"slot_id": "top", "display_name": "Tops"}]
        products: A list of products with metadata. Each product must have 'id', 'name', 
            'category', 'price', 'rating', 'score', and a 'facets' dictionary of attributes.
            Example: [{"id": "p1", "name": "Shirt", "category": "tops", "price": 25.0, "facets": {"color": "red"}}]
        rules: A list of structured constraint rules. Each rule can define:
            - 'category_constraints': List of {"slot_id": ..., "category": ...}
            - 'min_rating_constraints': List of {"slot_id": ..., "min_rating": ...}
            - 'price_constraints': {"max_total_price": ..., "max_slot_price": {"slot_id": price}}
            - 'compatibility_constraints': List of {"facet_name": ..., "match_type": "exact"}
            - 'required_product_ids': List of product IDs
            - 'excluded_product_ids': List of product IDs

    Returns:
        A dictionary containing valid bundle solutions sorted by total score and price.
    """
    # Parse dict inputs back into Pydantic models
    parsed_slots = [BundleSlot.model_validate(s) for s in slots]
    parsed_products = [BundleProduct.model_validate(p) for p in products]
    parsed_rules = [BundleRule.model_validate(r) for r in rules]

    # Instantiate the Constraint Satisfaction Problem (CSP) solver
    solver = BundleSolver(parsed_slots, parsed_products, parsed_rules)
    result = solver.solve()

    return result.model_dump()
