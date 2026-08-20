import pytest
from constructor_agent_tools.bundle.schemas.schemas import (
    BundleProduct,
    BundleRule,
    BundleSlot,
    CategoryConstraint,
    PriceConstraint,
    CompatibilityConstraint,
)
from constructor_agent_tools.bundle.solver import BundleSolver

def test_bundle_solver_simple():
    # 1. Setup slots
    slots = [
        BundleSlot(slot_id="top", display_name="Top Wear"),
        BundleSlot(slot_id="bottom", display_name="Bottom Wear")
    ]

    # 2. Setup products
    products = [
        BundleProduct(id="t1", name="Red T-shirt", category="tops", price=20.0, facets={"color": "red"}),
        BundleProduct(id="t2", name="Blue T-shirt", category="tops", price=25.0, facets={"color": "blue"}),
        BundleProduct(id="b1", name="Red Shorts", category="bottoms", price=30.0, facets={"color": "red"}),
        BundleProduct(id="b2", name="Blue Jeans", category="bottoms", price=50.0, facets={"color": "blue"}),
    ]

    # 3. Setup rules
    rules = [
        BundleRule(
            rule_name="Category Slots",
            description="Assign categories to slots",
            category_constraints=[
                CategoryConstraint(slot_id="top", category="tops"),
                CategoryConstraint(slot_id="bottom", category="bottoms")
            ]
        ),
        BundleRule(
            rule_name="Max Bundle Price",
            description="Limit total price",
            price_constraints=[PriceConstraint(max_total_price=60.0)]
        ),
        BundleRule(
            rule_name="Color Matching Compatibility",
            description="Ensure slots match color facet exactly",
            compatibility_constraints=[
                CompatibilityConstraint(facet_name="color", match_type="exact")
            ]
        )
    ]

    # Solve
    solver = BundleSolver(slots, products, rules)
    res = solver.solve()

    assert res.total_found == 1
    sol = res.solutions[0]
    assert sol.products["top"].id == "t1"
    assert sol.products["bottom"].id == "b1"
    assert sol.total_price == 50.0
