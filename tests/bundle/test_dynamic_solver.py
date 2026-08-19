import pytest
import numpy as np
from pathlib import Path
import shutil

from constructor_agent_tools.bundle.dynamic_schemas import (
    CatalogProduct,
    IntentResult,
    ScoredCandidate,
    SlotRule,
)
from constructor_agent_tools.bundle.embedding_engine import VectorStore
from constructor_agent_tools.bundle.dynamic_solver import DynamicBundleSolver


# --- VectorStore Tests ---

class TestVectorStore:
    def setup_method(self):
        """Create a temp directory for each test."""
        self.store_dir = Path("tests/bundle/_test_vector_store")
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up temp directory after each test."""
        if self.store_dir.exists():
            shutil.rmtree(self.store_dir)

    def test_add_and_search(self):
        store = VectorStore(store_dir=self.store_dir)

        # Add 3 products with simple 3D vectors
        store.add("phone_1", [1.0, 0.0, 0.0], {"title": "Pixel 8", "category": "phones"})
        store.add("charger_1", [0.9, 0.1, 0.0], {"title": "USB-C Charger", "category": "chargers"})
        store.add("case_1", [0.0, 0.0, 1.0], {"title": "Phone Case", "category": "cases"})

        # Search with a query close to phone/charger direction
        results = store.search([1.0, 0.0, 0.0], top_k=2)

        assert len(results) == 2
        assert results[0][0] == "phone_1"  # Most similar
        assert results[1][0] == "charger_1"  # Second most similar
        assert results[0][1] > results[1][1]  # Score ordering

    def test_category_filter(self):
        store = VectorStore(store_dir=self.store_dir)

        store.add("p1", [1.0, 0.0, 0.0], {"title": "Pixel 8", "category": "phones"})
        store.add("c1", [0.95, 0.05, 0.0], {"title": "USB-C Charger", "category": "chargers"})
        store.add("c2", [0.9, 0.1, 0.0], {"title": "Lightning Charger", "category": "chargers"})

        # Without filter: phone ranks first
        results_all = store.search([1.0, 0.0, 0.0], top_k=3)
        assert results_all[0][0] == "p1"

        # With category filter: only chargers
        results_filtered = store.search([1.0, 0.0, 0.0], top_k=3, category_filter="chargers")
        assert all(r[2].get("category") == "chargers" for r in results_filtered)
        assert results_filtered[0][0] == "c1"

    def test_save_and_load(self):
        store = VectorStore(store_dir=self.store_dir)
        store.add("p1", [1.0, 0.0, 0.0], {"title": "Pixel 8", "category": "phones"})
        store.add("c1", [0.9, 0.1, 0.0], {"title": "USB-C Charger", "category": "chargers"})
        store.save()

        # Load into a fresh store
        store2 = VectorStore(store_dir=self.store_dir)
        store2.load()

        assert store2.size == 2
        results = store2.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0] == "p1"

    def test_empty_store_search(self):
        store = VectorStore(store_dir=self.store_dir)
        results = store.search([1.0, 0.0, 0.0], top_k=5)
        assert results == []


# --- DynamicBundleSolver Tests ---

class TestDynamicBundleSolver:
    def test_solve_basic(self):
        """Test solver picks the best scoring products per slot under a price constraint."""
        intent = IntentResult(
            intent_type="hard",
            max_total_price=100.0,
            slots=[
                SlotRule(slot_id="charger", display_name="Charger", rule_description="USB-C charger"),
                SlotRule(slot_id="case", display_name="Case", rule_description="Phone case"),
            ],
        )

        candidates_per_slot = {
            "charger": [
                ScoredCandidate(
                    product=CatalogProduct(id="c1", title="30W USB-C Charger", price=25.0),
                    similarity_score=0.95,
                ),
                ScoredCandidate(
                    product=CatalogProduct(id="c2", title="65W USB-C Charger", price=45.0),
                    similarity_score=0.90,
                ),
            ],
            "case": [
                ScoredCandidate(
                    product=CatalogProduct(id="cs1", title="Slim Clear Case", price=15.0),
                    similarity_score=0.88,
                ),
                ScoredCandidate(
                    product=CatalogProduct(id="cs2", title="Rugged Armor Case", price=30.0),
                    similarity_score=0.85,
                ),
            ],
        }

        # We don't need a real embedding engine for solver-only tests
        solver = DynamicBundleSolver(
            embedding_engine=None,
            vector_store=None,
            top_k_per_slot=50,
        )

        solutions = solver.solve(intent, candidates_per_slot)

        # Should find valid bundles
        assert len(solutions) > 0

        # Best solution should have highest aggregate similarity
        best = solutions[0]
        assert best.total_price <= 100.0
        assert "charger" in best.slot_products
        assert "case" in best.slot_products

    def test_solve_no_duplicate_products(self):
        """Ensure the same product cannot fill two different slots."""
        intent = IntentResult(
            intent_type="soft",
            slots=[
                SlotRule(slot_id="slot_a", display_name="A", rule_description="test"),
                SlotRule(slot_id="slot_b", display_name="B", rule_description="test"),
            ],
        )

        shared_product = CatalogProduct(id="shared_1", title="Shared Product", price=10.0)

        candidates_per_slot = {
            "slot_a": [
                ScoredCandidate(product=shared_product, similarity_score=0.99),
            ],
            "slot_b": [
                ScoredCandidate(product=shared_product, similarity_score=0.99),
                ScoredCandidate(
                    product=CatalogProduct(id="other_1", title="Other", price=10.0),
                    similarity_score=0.80,
                ),
            ],
        }

        solver = DynamicBundleSolver(embedding_engine=None, vector_store=None)
        solutions = solver.solve(intent, candidates_per_slot)

        # All solutions must have distinct product IDs across slots
        for sol in solutions:
            ids = [p.id for p in sol.slot_products.values()]
            assert len(ids) == len(set(ids)), "Duplicate product found across slots"

    def test_solve_price_constraint(self):
        """Bundles exceeding max_total_price must be excluded."""
        intent = IntentResult(
            intent_type="hard",
            max_total_price=50.0,
            slots=[
                SlotRule(slot_id="a", display_name="A", rule_description="test"),
                SlotRule(slot_id="b", display_name="B", rule_description="test"),
            ],
        )

        candidates_per_slot = {
            "a": [
                ScoredCandidate(
                    product=CatalogProduct(id="a1", title="Cheap", price=20.0),
                    similarity_score=0.9,
                ),
                ScoredCandidate(
                    product=CatalogProduct(id="a2", title="Expensive", price=40.0),
                    similarity_score=0.95,
                ),
            ],
            "b": [
                ScoredCandidate(
                    product=CatalogProduct(id="b1", title="Cheap B", price=20.0),
                    similarity_score=0.85,
                ),
            ],
        }

        solver = DynamicBundleSolver(embedding_engine=None, vector_store=None)
        solutions = solver.solve(intent, candidates_per_slot)

        for sol in solutions:
            assert sol.total_price <= 50.0, f"Bundle exceeded price cap: {sol.total_price}"
