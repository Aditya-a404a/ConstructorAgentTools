# Constructor Agent Tools

A suite of intelligent decision engines and API interfaces designed to optimize search discovery, product rankings, and recommendation bundles for e-commerce.

---

## Architecture Overview

This project separates human intent understanding (handled by low-latency LLMs) from high-precision, deterministic optimization algorithms (handled by local, non-API-based graph search and optimization solvers).

```
               [ Merchandiser Request ]
                           │
                           ▼
             [ Gemini 2.5 flash Generator ]
            (Classifies intent & constraints)
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
   [ Product Bundling ]           [ Searchandising ]
   (Graph CSP Solver)            (QUBO Optimization)
  DFS Backtracking search        Simulated Annealing
            │                             │
            └──────────────┬──────────────┘
                           ▼
                 [ Sorted Results JSON ]
```

---

## 1. Product Bundling Engine (`src/constructor_agent_tools/bundle`)

Given an anchor product and a natural language merchandising request (e.g., *"Create an outfit with a matching color top, bottom, and shoes under $150"*), the bundler recommends a set of highly compatible products.

*   **Rule Generator (`generator.py`)**: Translates merchandiser instructions into structured `BundleRule` constraints (category restrictions, rating thresholds, budget limits, slot price caps, and compatibility facets).
*   **CSP Solver (`solver.py`)**: Uses a **Constraint Satisfaction Problem (CSP)** solver built with a **Backtracking Depth-First Search (DFS)** algorithm:
    *   **MRV (Minimum Remaining Values) Heuristic**: Dynamically sorts slots to solve the most constrained ones first, pruning invalid paths early.
    *   **Unary & Binary Pruning**: Instantly filters products by category, rating, exclusions, and individual slot prices before search.
    *   **Timeout Safety**: Constrained by `BUNDLE_SOLVER_TIMEOUT_SECONDS` settings to guarantee sub-100ms response times.
*   **Agent Tool (`tools.py`)**: Exposes the solver as a clean python function wrapper `find_best_bundles` suitable for Gemini function calling.

---

## 2. Searchandising QUBO Optimizer (`src/constructor_agent_tools/searchandising`)

Optimizes search result layouts by balancing three competing objectives: search relevance, business metrics (profit margin, inventory stock, sponsored bids), and layout diversity.

*   **QUBO Matrix Formulation (`qubo_solver.py`)**: Formulates search ranking as a **Quadratic Unconstrained Binary Optimization (QUBO)** model:
    $$H(x) = -\sum_{i} (\alpha \cdot \text{Relevance}_i + \beta \cdot \text{BusinessScore}_i) x_i + \gamma \sum_{i \neq j} \text{Similarity}(i, j) x_i x_j + P(\sum_i x_i - K)^2$$
    *   **Linear Terms**: Promotes relevant and highly profitable items.
    *   **Quadratic Terms**: Penalizes choosing pairs of similar items (using Jaccard similarity on product facets) to increase layout diversity.
    *   **Penalty constraints**: Enforces selecting exactly $K$ slots.
*   **Annealing Solver (`qubo_solver.py`)**: Solves the QUBO model locally using a **Simulated Annealing** algorithm with an exponential temperature schedule and Metropolis acceptance criteria.
*   **Agent Tool (`tools.py`)**: Exposes the solver as `optimize_search_ranking` for agent function calling.

---

## Setup & Quickstart

### Prerequisites
*   Python 3.9+
*   Gemini API Key (set in environment or `.env`)

### Installation & Run

1.  **Clone and create virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies in edit/development mode**:
    ```bash
    pip install -e .
    ```

3.  **Add your Gemini API Key to `.env`**:
    ```env
    GEMINI_API_KEY="your-api-key-here"
    ```

4.  **Start the server**:
    ```bash
    python -m constructor_agent_tools
    ```

5.  **Verify server status**:
    ```bash
    curl http://localhost:8001/health
    ```

---

## API Documentation

### Product Bundling API
*   `POST /bundle/generate-rules`: Submits merchandiser natural language constraints and slots, returning a structured list of constraints.
*   `POST /bundle/solve`: Inputs product candidates, slots, and rules, returning ranked valid bundle configurations.

### Searchandising API
*   `POST /searchandising/generate-params`: Translates merchant instructions into QUBO weight coefficients ($\alpha, \beta, \gamma$).
*   `POST /searchandising/optimize`: Solves the QUBO ranking problem using simulated annealing for a candidate set of products.
