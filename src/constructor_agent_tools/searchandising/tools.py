from typing import Any, Dict, List

from constructor_agent_tools.searchandising.schemas import (
    MerchandisingParams,
    SearchProduct,
)
from constructor_agent_tools.searchandising.qubo_solver import QUBOSolver


def optimize_search_ranking(
    products: List[Dict[str, Any]],
    params: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Optimizes the ranking of search results using a QUBO (Quadratic Unconstrained 
    Binary Optimization) model solved via simulated annealing.

    Balances three competing objectives:
    - Relevance: How well products match the user's search query.
    - Business value: Profit margin, inventory levels, and sponsorship status.
    - Diversity: Penalizes showing highly similar products next to each other.

    Args:
        products: A list of product dictionaries. Each product must have 'id', 'title',
            and should include 'relevance_score', 'margin', 'inventory', 'is_sponsored',
            'category', and 'facets' for optimal results.
            Example: [{"id": "p1", "title": "Nike Air Max", "relevance_score": 0.95, 
                       "margin": 45.0, "category": "running_shoes", 
                       "facets": {"brand": "Nike", "color": "black"}}]
        params: Optional dictionary of QUBO weight parameters:
            - alpha (float): Relevance weight (default 1.0)
            - beta (float): Business boost weight (default 0.5)
            - gamma (float): Diversity penalty weight (default 0.3)
            - num_slots (int): Number of products to select (default 10)

    Returns:
        A dictionary containing the optimized ranking with per-product score breakdowns.
    """
    parsed_products = [SearchProduct.model_validate(p) for p in products]
    parsed_params = MerchandisingParams.model_validate(params) if params else MerchandisingParams()

    solver = QUBOSolver()
    result = solver.solve(parsed_products, parsed_params)

    return result.model_dump()
