import pytest
from constructor_agent_tools.searchandising.schemas import (
    MerchandisingParams,
    SearchProduct,
)
from constructor_agent_tools.searchandising.qubo_solver import QUBOSolver


def _make_product(id: str, title: str, category: str, relevance: float,
                  margin: float = 20.0, facets: dict = None) -> SearchProduct:
    """Helper to create test products."""
    return SearchProduct(
        id=id, title=title, category=category,
        relevance_score=relevance, margin=margin,
        facets=facets or {},
    )


class TestQUBOSolver:

    def test_selects_correct_number_of_slots(self):
        """Solver must select exactly num_slots products."""
        products = [
            _make_product("p1", "Shoe A", "shoes", 0.9, facets={"brand": "Nike"}),
            _make_product("p2", "Shoe B", "shoes", 0.8, facets={"brand": "Adidas"}),
            _make_product("p3", "Shoe C", "shoes", 0.7, facets={"brand": "Puma"}),
            _make_product("p4", "Shoe D", "shoes", 0.6, facets={"brand": "Reebok"}),
            _make_product("p5", "Shoe E", "shoes", 0.5, facets={"brand": "NB"}),
        ]
        params = MerchandisingParams(num_slots=3)

        solver = QUBOSolver()
        result = solver.solve(products, params)

        assert result.num_selected == 3

    def test_relevance_dominates_when_alpha_high(self):
        """With high alpha, the most relevant products should be selected."""
        products = [
            _make_product("high_rel", "Top Match", "shoes", 0.99, margin=5.0,
                          facets={"brand": "X"}),
            _make_product("low_rel_high_margin", "Profit King", "shoes", 0.1, margin=95.0,
                          facets={"brand": "Y"}),
            _make_product("mid", "Decent", "shoes", 0.5, margin=50.0,
                          facets={"brand": "Z"}),
        ]
        params = MerchandisingParams(alpha=5.0, beta=0.01, gamma=0.0, num_slots=1)

        solver = QUBOSolver()
        result = solver.solve(products, params)

        assert result.ranked_products[0].product.id == "high_rel"

    def test_business_boost_when_beta_high(self):
        """With high beta, high-margin products should be favored."""
        products = [
            _make_product("low_margin", "Cheap Item", "shoes", 0.9, margin=5.0,
                          facets={"brand": "A"}),
            _make_product("high_margin", "Premium Item", "shoes", 0.9, margin=95.0,
                          facets={"brand": "B"}),
            _make_product("mid_margin", "Mid Item", "shoes", 0.9, margin=50.0,
                          facets={"brand": "C"}),
        ]
        params = MerchandisingParams(alpha=0.01, beta=5.0, gamma=0.0, num_slots=1)

        solver = QUBOSolver()
        result = solver.solve(products, params)

        assert result.ranked_products[0].product.id == "high_margin"

    def test_diversity_penalty_avoids_duplicates(self):
        """With high gamma, solver should avoid selecting very similar products."""
        # 3 identical Nike shoes (same facets) and 1 different Adidas shoe
        products = [
            _make_product("nike1", "Nike Air 1", "shoes", 0.9, facets={"brand": "Nike", "color": "black"}),
            _make_product("nike2", "Nike Air 2", "shoes", 0.9, facets={"brand": "Nike", "color": "black"}),
            _make_product("nike3", "Nike Air 3", "shoes", 0.9, facets={"brand": "Nike", "color": "black"}),
            _make_product("adidas1", "Adidas Ultra", "shoes", 0.85, facets={"brand": "Adidas", "color": "white"}),
        ]
        # High diversity penalty
        params = MerchandisingParams(alpha=1.0, beta=0.0, gamma=5.0, num_slots=2)

        solver = QUBOSolver()
        result = solver.solve(products, params)

        selected_ids = {r.product.id for r in result.ranked_products}
        # The Adidas shoe should be selected because picking 2 identical Nikes
        # incurs a massive diversity penalty
        assert "adidas1" in selected_ids

    def test_empty_products(self):
        """Solver should handle empty product list gracefully."""
        solver = QUBOSolver()
        result = solver.solve([], MerchandisingParams())

        assert result.num_selected == 0
        assert result.ranked_products == []

    def test_fewer_products_than_slots(self):
        """When fewer products exist than slots, return all products."""
        products = [
            _make_product("p1", "Only Shoe", "shoes", 0.9),
        ]
        params = MerchandisingParams(num_slots=5)

        solver = QUBOSolver()
        result = solver.solve(products, params)

        assert result.num_selected == 1

    def test_similarity_calculation(self):
        """Test the Jaccard similarity function directly."""
        p1 = _make_product("a", "A", "shoes", 1.0, facets={"brand": "Nike", "color": "black"})
        p2 = _make_product("b", "B", "shoes", 1.0, facets={"brand": "Nike", "color": "black"})
        p3 = _make_product("c", "C", "boots", 1.0, facets={"brand": "Adidas", "color": "white"})

        # Identical facets + same category → very high similarity
        sim_same = QUBOSolver._compute_similarity(p1, p2)
        assert sim_same > 0.9

        # Different facets + different category → low similarity
        sim_diff = QUBOSolver._compute_similarity(p1, p3)
        assert sim_diff < 0.3
