import time
import asyncio
from typing import Dict, List, Set, Tuple
from constructor_agent_tools.settings import settings
from constructor_agent_tools.bundle.schemas.schemas import (
    BundleProduct,
    BundleRule,
    BundleSlot,
    BundleSolution,
    BundleSolutionResponse,
)

class BundleSolver:
    """
    Encodes the product catalog and slot requirements as a Multipartite Graph.
    Uses backtracking/DFS search with pruning and heuristics to find valid bundles.
    """
    def __init__(self, slots: List[BundleSlot], products: List[BundleProduct], rules: List[BundleRule]):
        self.slots = slots
        self.products = products
        self.rules = rules
        self.timeout = settings.BUNDLE_SOLVER_TIMEOUT_SECONDS

        # Group products by category mapping
        self.category_to_products: Dict[str, List[BundleProduct]] = {}
        for p in products:
            self.category_to_products.setdefault(p.category, []).append(p)
            
        self.product_map: Dict[str, BundleProduct] = {p.id: p for p in products}

    def solve(self) -> BundleSolutionResponse:
        start_time = time.time()
        
        # 1. Determine candidates for each slot
        slot_candidates: Dict[str, List[BundleProduct]] = {}
        
        # Build category map for slots from rules
        slot_categories: Dict[str, Set[str]] = {}
        for rule in self.rules:
            for cc in rule.category_constraints:
                slot_categories.setdefault(cc.slot_id, set()).add(cc.category)

        for slot in self.slots:
            sid = slot.slot_id
            cats = slot_categories.get(sid, set())
            
            candidates = []
            if not cats:
                # If no category constraints, all products are eligible
                candidates = self.products
            else:
                for cat in cats:
                    candidates.extend(self.category_to_products.get(cat, []))
            
            # Apply individual product rules (ratings, slot price limit, exclusions)
            filtered_candidates = []
            for p in candidates:
                # Exclusion constraint
                excluded = False
                for rule in self.rules:
                    if p.id in rule.excluded_product_ids:
                        excluded = True
                        break
                if excluded:
                    continue

                # Rating constraint
                rating_ok = True
                for rule in self.rules:
                    for mr in rule.min_rating_constraints:
                        if (mr.slot_id == sid or mr.slot_id is None) and p.rating < mr.min_rating:
                            rating_ok = False
                            break
                if not rating_ok:
                    continue

                # Price constraint
                price_ok = True
                for rule in self.rules:
                    if rule.price_constraints and sid in rule.price_constraints.max_slot_price:
                        if p.price > rule.price_constraints.max_slot_price[sid]:
                            price_ok = False
                            break
                if not price_ok:
                    continue

                filtered_candidates.append(p)
            
            slot_candidates[sid] = filtered_candidates

        # 2. Backtracking DFS solver to search for valid combinations across slots
        solutions: List[BundleSolution] = []
        current_solution: Dict[str, BundleProduct] = {}
        
        # Sort slots by number of candidates (MRV heuristic - Minimum Remaining Values)
        sorted_slots = sorted(self.slots, key=lambda s: len(slot_candidates.get(s.slot_id, [])))
        
        def backtrack(slot_idx: int):
            # Check timeout
            if time.time() - start_time > self.timeout:
                return

            if slot_idx == len(sorted_slots):
                # We have a candidate assignment for every slot! Now check bundle-wide constraints.
                # Max total price constraint
                total_price = sum(p.price for p in current_solution.values())
                for rule in self.rules:
                    if rule.price_constraints and rule.price_constraints.max_total_price is not None:
                        if total_price > rule.price_constraints.max_total_price:
                            return
                
                # Check compatibility constraints (exact facet matching)
                for rule in self.rules:
                    for compat in rule.compatibility_constraints:
                        facet_val = None
                        for p in current_solution.values():
                            val = p.facets.get(compat.facet_name)
                            if val is None:
                                return  # Missing facet means not compatible
                            if facet_val is None:
                                facet_val = val
                            elif facet_val != val:
                                return  # Mismatch

                # Check required product constraints
                for rule in self.rules:
                    for req_id in rule.required_product_ids:
                        if req_id not in [p.id for p in current_solution.values()]:
                            return

                # Valid solution found!
                avg_rating = sum(p.rating for p in current_solution.values()) / len(current_solution)
                total_score = sum(p.score for p in current_solution.values())
                solutions.append(BundleSolution(
                    products=dict(current_solution),
                    total_price=round(total_price, 2),
                    average_rating=round(avg_rating, 2),
                    total_score=round(total_score, 2)
                ))
                return

            slot = sorted_slots[slot_idx]
            sid = slot.slot_id
            
            for product in slot_candidates[sid]:
                # Distinct check: Don't pick the same product for two different slots
                if product.id in [p.id for p in current_solution.values()]:
                    continue
                
                current_solution[sid] = product
                backtrack(slot_idx + 1)
                del current_solution[sid]

        # Start recursion
        if sorted_slots:
            backtrack(0)

        # Sort solutions by total score (relevance) descending, and then price ascending
        solutions.sort(key=lambda s: (-s.total_score, s.total_price))

        return BundleSolutionResponse(
            solutions=solutions,
            total_found=len(solutions)
        )
