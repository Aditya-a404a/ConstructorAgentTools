import logging
import math
import random
import time
from typing import Dict, List, Set, Tuple

from constructor_agent_tools.settings import settings
from constructor_agent_tools.searchandising.schemas import (
    MerchandisingParams,
    OptimizeRankingResponse,
    RankedProduct,
    SearchProduct,
)

logger = logging.getLogger("constructor_agent_tools.searchandising.qubo_solver")


class QUBOSolver:
    """
    Builds and solves a QUBO (Quadratic Unconstrained Binary Optimization) model
    for product ranking optimization.

    Objective function to MINIMIZE:
        H(x) = - Σ_i  h_i * x_i  +  Σ_{i<j}  J_{ij} * x_i * x_j  +  P * (Σ_i x_i - K)^2

    Where:
        h_i = α * relevance_i + β * business_score_i   (linear: reward for selecting product i)
        J_ij = γ * similarity(i, j)                     (quadratic: penalty for selecting similar pair)
        P * (Σ x_i - K)^2                              (penalty: forces exactly K products to be selected)
        K = num_slots                                   (number of products to pick)
    """

    def __init__(self, timeout: float = None):
        self.timeout = timeout or settings.BUNDLE_SOLVER_TIMEOUT_SECONDS

    # ─── Similarity Calculation ───────────────────────────────────────────

    @staticmethod
    def _compute_similarity(p1: SearchProduct, p2: SearchProduct) -> float:
        """
        Compute pairwise similarity between two products based on shared facets.
        Returns a value in [0, 1] where 1 means identical facets.
        Uses Jaccard similarity on facet key-value pairs.
        """
        if not p1.facets and not p2.facets:
            # If no facets, fall back to category match
            return 1.0 if p1.category == p2.category else 0.0

        set1 = {f"{k}={v}" for k, v in p1.facets.items()}
        set2 = {f"{k}={v}" for k, v in p2.facets.items()}

        if not set1 and not set2:
            return 1.0 if p1.category == p2.category else 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        jaccard = intersection / union

        # Add category bonus: same category adds extra similarity
        if p1.category == p2.category:
            jaccard = min(1.0, jaccard + 0.3)

        return jaccard

    @staticmethod
    def _compute_business_score(product: SearchProduct) -> float:
        """
        Compute a normalized business score from margin, inventory, and sponsorship.
        """
        score = 0.0

        # Margin contribution (normalized to ~0-1 range assuming margin is a percentage)
        score += min(product.margin / 100.0, 1.0) * 0.5

        # Inventory contribution (high stock = push to sell)
        if product.inventory > 0:
            score += min(product.inventory / 1000.0, 1.0) * 0.3

        # Sponsorship bonus
        if product.is_sponsored:
            score += 0.2

        return score

    # ─── QUBO Matrix Construction ─────────────────────────────────────────

    def build_qubo_matrix(
        self,
        products: List[SearchProduct],
        params: MerchandisingParams,
    ) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
        """
        Build the QUBO model coefficients.

        Returns:
            linear: Dict[i, h_i] — linear coefficients (reward for selecting product i)
            quadratic: Dict[(i,j), J_ij] — quadratic coefficients (penalty for selecting pair)
            penalty_weight: The constraint penalty weight P for enforcing num_slots selection
        """
        n = len(products)
        alpha, beta, gamma = params.alpha, params.beta, params.gamma
        K = params.num_slots

        # --- Linear terms: h_i = α * relevance + β * business ---
        linear: Dict[int, float] = {}
        for i, product in enumerate(products):
            biz_score = self._compute_business_score(product)
            h_i = alpha * product.relevance_score + beta * biz_score
            linear[i] = h_i

        # --- Quadratic terms: J_ij = γ * similarity(i, j) ---
        quadratic: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._compute_similarity(products[i], products[j])
                if sim > 0.01:  # Skip near-zero similarity pairs
                    quadratic[(i, j)] = gamma * sim

        # --- Penalty weight for constraint Σ x_i = K ---
        # P must be large enough that violating the constraint is never worth it
        max_h = max(linear.values()) if linear else 1.0
        penalty_weight = 2.0 * max_h * n  # Scaled to dominate any linear gain

        return linear, quadratic, penalty_weight

    # ─── QUBO Energy Function ─────────────────────────────────────────────

    @staticmethod
    def _compute_energy(
        state: List[int],
        linear: Dict[int, float],
        quadratic: Dict[Tuple[int, int], float],
        penalty_weight: float,
        target_k: int,
    ) -> float:
        """
        Compute the QUBO objective value H(x) for a given binary state vector.

        H(x) = -Σ h_i * x_i  +  Σ J_ij * x_i * x_j  +  P * (Σ x_i - K)^2
        """
        energy = 0.0

        # Linear: reward for each selected product (negative = lower energy = better)
        for i, x_i in enumerate(state):
            if x_i == 1:
                energy -= linear.get(i, 0.0)

        # Quadratic: penalty for co-selected similar products
        for (i, j), j_ij in quadratic.items():
            if state[i] == 1 and state[j] == 1:
                energy += j_ij

        # Constraint penalty: punish deviations from exactly K selected
        selected_count = sum(state)
        energy += penalty_weight * (selected_count - target_k) ** 2

        return energy

    # ─── Simulated Annealing Solver ───────────────────────────────────────

    def solve(
        self,
        products: List[SearchProduct],
        params: MerchandisingParams,
    ) -> OptimizeRankingResponse:
        """
        Solve the QUBO ranking optimization using Simulated Annealing.
        
        Returns the optimal selection and ordering of products.
        """
        start_time = time.time()
        n = len(products)
        K = min(params.num_slots, n)  # Can't select more than available

        if n == 0:
            return OptimizeRankingResponse(
                ranked_products=[], total_objective_value=0.0,
                num_candidates=0, num_selected=0,
            )

        # If we have fewer products than slots, just return all of them
        if n <= K:
            ranked = []
            for product in products:
                biz = self._compute_business_score(product)
                ranked.append(RankedProduct(
                    product=product,
                    relevance_contribution=params.alpha * product.relevance_score,
                    business_contribution=params.beta * biz,
                    diversity_penalty=0.0,
                    final_score=params.alpha * product.relevance_score + params.beta * biz,
                ))
            ranked.sort(key=lambda r: -r.final_score)
            return OptimizeRankingResponse(
                ranked_products=ranked, total_objective_value=0.0,
                num_candidates=n, num_selected=n,
            )

        # Build QUBO matrix
        linear, quadratic, penalty_weight = self.build_qubo_matrix(products, params)

        # --- Simulated Annealing ---
        # Initialize: greedily select top-K by linear score
        scored_indices = sorted(range(n), key=lambda i: -linear.get(i, 0.0))
        state = [0] * n
        for i in scored_indices[:K]:
            state[i] = 1

        current_energy = self._compute_energy(state, linear, quadratic, penalty_weight, K)
        best_state = list(state)
        best_energy = current_energy

        # Annealing schedule
        T_start = 2.0
        T_end = 0.01
        max_iterations = min(n * n * 10, 50000)  # Scale with problem size, cap at 50K

        for iteration in range(max_iterations):
            # Check timeout
            if time.time() - start_time > self.timeout:
                logger.warning(f"QUBO solver timed out after {iteration} iterations")
                break

            # Temperature schedule (exponential decay)
            t = iteration / max_iterations
            T = T_start * ((T_end / T_start) ** t)

            # Propose a swap: flip one selected OFF and one unselected ON
            selected = [i for i in range(n) if state[i] == 1]
            unselected = [i for i in range(n) if state[i] == 0]

            if not selected or not unselected:
                break

            flip_off = random.choice(selected)
            flip_on = random.choice(unselected)

            # Apply swap
            state[flip_off] = 0
            state[flip_on] = 1

            new_energy = self._compute_energy(state, linear, quadratic, penalty_weight, K)
            delta = new_energy - current_energy

            # Metropolis acceptance criterion
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                # Accept the new state
                current_energy = new_energy
                if new_energy < best_energy:
                    best_state = list(state)
                    best_energy = new_energy
            else:
                # Reject: undo swap
                state[flip_off] = 1
                state[flip_on] = 0

        # --- Build Response ---
        selected_indices = [i for i in range(n) if best_state[i] == 1]

        ranked_products: List[RankedProduct] = []
        for i in selected_indices:
            product = products[i]
            biz_score = self._compute_business_score(product)
            relevance_contrib = params.alpha * product.relevance_score
            business_contrib = params.beta * biz_score

            # Compute this product's diversity penalty with other selected products
            div_penalty = 0.0
            for j in selected_indices:
                if i != j:
                    key = (min(i, j), max(i, j))
                    div_penalty += quadratic.get(key, 0.0)

            final = relevance_contrib + business_contrib - div_penalty

            ranked_products.append(RankedProduct(
                product=product,
                relevance_contribution=round(relevance_contrib, 4),
                business_contribution=round(business_contrib, 4),
                diversity_penalty=round(div_penalty, 4),
                final_score=round(final, 4),
            ))

        # Sort by final score descending
        ranked_products.sort(key=lambda r: -r.final_score)

        elapsed = time.time() - start_time
        logger.info(f"QUBO solver completed in {elapsed:.2f}s, selected {len(ranked_products)}/{n} products")

        return OptimizeRankingResponse(
            ranked_products=ranked_products,
            total_objective_value=round(best_energy, 4),
            num_candidates=n,
            num_selected=len(ranked_products),
        )
