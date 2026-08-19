import logging
import time
from typing import Dict, List, Optional

from constructor_agent_tools.settings import settings
from constructor_agent_tools.bundle.dynamic_schemas import (
    CatalogProduct,
    DynamicBundleSolution,
    IntentResult,
    ScoredCandidate,
)
from constructor_agent_tools.bundle.embedding_engine import EmbeddingEngine, VectorStore

logger = logging.getLogger("constructor_agent_tools.bundle.dynamic_solver")


class DynamicBundleSolver:
    """
    Orchestrates the full dynamic bundle generation flow:
    1. Takes the IntentResult (from Call 1).
    2. Embeds each slot's rule_description (Call 2).
    3. Runs vector search per slot to retrieve candidates.
    4. Solves for optimal bundles under global constraints (price, dedup).
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        top_k_per_slot: int = 50,
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.top_k_per_slot = top_k_per_slot
        self.timeout = settings.BUNDLE_SOLVER_TIMEOUT_SECONDS

    async def retrieve_candidates(self, intent: IntentResult) -> Dict[str, List[ScoredCandidate]]:
        """
        Embed each slot's rule_description and vector-search the catalog
        to retrieve candidate products per slot.
        """
        # Batch embed all rule descriptions in a single API call (Call 2)
        rule_texts = [slot.rule_description for slot in intent.slots]
        rule_vectors = await self.embedding_engine.embed_texts(rule_texts)

        candidates_per_slot: Dict[str, List[ScoredCandidate]] = {}

        for slot, rule_vec in zip(intent.slots, rule_vectors):
            # Vector search with optional category hint filtering
            results = self.vector_store.search(
                query_vector=rule_vec,
                top_k=self.top_k_per_slot,
                category_filter=slot.category_hint,
            )

            candidates = []
            for product_id, score, metadata in results:
                product = CatalogProduct(
                    id=metadata.get("id", product_id),
                    title=metadata.get("title", ""),
                    description=metadata.get("description", ""),
                    category=metadata.get("category", ""),
                    price=metadata.get("price", 0.0),
                    rating=metadata.get("rating", 5.0),
                    image_url=metadata.get("image_url"),
                )
                candidates.append(ScoredCandidate(product=product, similarity_score=score))

            candidates_per_slot[slot.slot_id] = candidates
            logger.info(f"Slot '{slot.slot_id}': retrieved {len(candidates)} candidates")

        return candidates_per_slot

    def solve(
        self,
        intent: IntentResult,
        candidates_per_slot: Dict[str, List[ScoredCandidate]],
    ) -> List[DynamicBundleSolution]:
        """
        Given candidates per slot, find the best bundle combinations
        that satisfy global constraints (price cap, no duplicate products).
        Uses DFS backtracking with timeout.
        """
        start_time = time.time()
        solutions: List[DynamicBundleSolution] = []

        slots = intent.slots
        max_price = intent.max_total_price

        # Working state for backtracking
        current_products: Dict[str, CatalogProduct] = {}
        current_scores: Dict[str, float] = {}
        used_ids: set = set()

        def backtrack(slot_idx: int):
            if time.time() - start_time > self.timeout:
                return

            if slot_idx == len(slots):
                # All slots filled — check global constraints
                total_price = sum(p.price for p in current_products.values())
                if max_price is not None and total_price > max_price:
                    return

                aggregate = sum(current_scores.values()) / len(current_scores)
                solutions.append(DynamicBundleSolution(
                    slot_products=dict(current_products),
                    slot_scores=dict(current_scores),
                    total_price=round(total_price, 2),
                    aggregate_similarity=round(aggregate, 4),
                ))
                return

            slot = slots[slot_idx]
            candidates = candidates_per_slot.get(slot.slot_id, [])

            for candidate in candidates:
                # No duplicate products across slots
                if candidate.product.id in used_ids:
                    continue

                current_products[slot.slot_id] = candidate.product
                current_scores[slot.slot_id] = candidate.similarity_score
                used_ids.add(candidate.product.id)

                backtrack(slot_idx + 1)

                del current_products[slot.slot_id]
                del current_scores[slot.slot_id]
                used_ids.discard(candidate.product.id)

        backtrack(0)

        # Sort by aggregate similarity descending, then price ascending
        solutions.sort(key=lambda s: (-s.aggregate_similarity, s.total_price))

        logger.info(
            f"Solver found {len(solutions)} solutions in "
            f"{time.time() - start_time:.2f}s"
        )
        return solutions

    async def generate_bundle(self, intent: IntentResult) -> List[DynamicBundleSolution]:
        """
        End-to-end: retrieve candidates via vector search, then solve.
        """
        candidates = await self.retrieve_candidates(intent)
        return self.solve(intent, candidates)
