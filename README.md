# ⚡ Constructor Agent Tools

> **Autonomous AI Decision Engines & Mathematical Solvers for Next-Gen E-Commerce Merchandising**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-8E75B2.svg?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 The Problem: Why Constructor Agent Tools?

Modern e-commerce merchandising faces two major engineering bottlenecks:

1. **The Combinatorial Explosion of Product Bundling (CSP)**:
   Creating multi-item bundles (e.g., *"An iPhone 17 Pro with a matching case, screen protector, and charger under $1100"*) requires checking strict compatibility, price ceilings, ratings, and category alignment across thousands of SKUs. Hand-curating rules doesn't scale, and pure LLMs hallucinate non-existent product IDs or fail at strict arithmetic constraints.

2. **The Multi-Objective Conflict of Search Ranking (QUBO)**:
   Merchandisers constantly struggle to balance three competing objectives on search result pages:
   - **Customer Relevance**: Showing what the user actually searched for.
   - **Business Profitability**: Boosting high-margin, overstocked, or sponsored products.
   - **Catalog Diversity**: Preventing a search for *"shoes"* from showing a wall of 15 identical black sneakers.

---

## 💡 The Core Idea: Hybrid Neuro-Symbolic Architecture

**Constructor Agent Tools** bridges the gap between **Generative AI** and **Deterministic Mathematical Optimization**:

* **Low-Latency LLMs (Gemini)** handle *natural language understanding, slot discovery, intent translation, and query extraction*.
* **Local Mathematical Solvers (CSP & QUBO)** handle *hard constraint enforcement, combinatorial search, and multi-objective ranking*.

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                 Merchant Instruction                    │
                              │    "Create an iPhone 17 Pro bundle under $1100"        │
                              └──────────────────────────┬──────────────────────────────┘
                                                         │
                                    ┌────────────────────┴───────────────────┐
                                    ▼                                        ▼
                   ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
                   │        1. Bundle Agent           │    │     2. Searchandising Agent      │
                   │   (CSP - Backtracking Solver)    │    │      (QUBO / Annealing Solver)   │
                   └────────────────┬─────────────────┘    └────────────────┬─────────────────┘
                                    │                                        │
                                    │ 1. Dynamic Slot Discovery              │ 1. Search Query Extraction
                                    │ 2. Sourcing (Mock Constructor)         │ 2. Catalog Sourcing & Metrics
                                    │ 3. LLM Rule Compilation               │ 3. LLM Weight Tuning (α, β, γ)
                                    │ 4. DFS Backtracking Solver (MRV)       │ 4. Simulated Annealing Matrix
                                    ▼                                        ▼
                   ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
                   │  100% Compatible Bundle Solution │    │    Optimized Product Ranking     │
                   └──────────────────────────────────┘    └──────────────────────────────────┘
```

---

## 🚀 Key Modules

### 1. Autonomous Product Bundling Engine (`src/constructor_agent_tools/bundle`)
* **Dynamic Slot Discovery**: Automatically identifies required bundle components from intent (e.g. `phone`, `case`, `charger`).
* **Automated Catalog Sourcing**: Automatically calls Constructor.io search endpoints to retrieve candidate items.
* **Constraint Satisfaction Problem (CSP) Solver**:
  * **MRV Heuristic (Minimum Remaining Values)**: Solves the most constrained slots first to prune exponential search spaces.
  * **Strict Constraint Verification**: Enforces total budget limits, slot price caps, minimum star ratings, and cross-item facet matching (e.g., `compatible_devices = "iPhone 17 Pro"`).
  * **Sub-100ms Performance**: Guaranteed sub-second execution with configurable timeout boundaries.

### 2. Searchandising QUBO Optimizer (`src/constructor_agent_tools/searchandising`)
* **Natural Language Parameter Generation**: Translates merchant goals (e.g., *"Push high margin Nike running shoes with a good variety of colors"*) into optimal mathematical coefficients:
  * **$\alpha$ (Alpha)**: Search query relevance weight.
  * **$\beta$ (Beta)**: Business metric weight (margin, inventory, sponsorship).
  * **$\gamma$ (Gamma)**: Layout diversity penalty weight.
* **QUBO Mathematical Formulation**:
  $$\min_{x} \left( -\alpha \sum_{i} \text{Relevance}_i x_i - \beta \sum_{i} \text{Margin}_i x_i + \gamma \sum_{i \neq j} \text{Similarity}(i, j) x_i x_j + P \left(\sum_i x_i - K\right)^2 \right)$$
* **Simulated Annealing Solver**: Uses Metropolis acceptance criteria with exponential cooling to find global minima for product slotting.

### 3. Mock Constructor Engine (`src/constructor_agent_tools/mock_constructor`)
* A dynamic, intelligent mock of Constructor.io's Search and Browse APIs that synthesizes realistic product catalogs, facets, prices, and categories on demand—allowing the entire toolchain to run standalone without external database dependencies.

---

## 🛠️ Quickstart Guide

### Prerequisites
* **Python 3.9+**
* **Google Gemini API Key** ([Get one here](https://aistudio.google.com/))

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/Aditya-a404a/ConstructorAgentTools.git
cd ConstructorAgentTools

python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -e .
```

### 3. Configure Environment
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### 4. Run the Backend Server
```bash
python -m constructor_agent_tools
```
* The FastAPI server will start on **`http://localhost:8000`** with CORS enabled.
* Interactive Swagger Docs: **`http://localhost:8000/docs`**
* Health check: **`http://localhost:8000/health`**

---

## 📡 API Endpoints

### 🛍️ End-to-End Agentic Flows (No pre-supplied products needed!)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/bundle/execute-flow` | Autonomous end-to-end bundling from a natural language prompt. |
| `POST` | `/searchandising/execute-flow` | Autonomous search query extraction, parameter tuning, and QUBO ranking. |

#### Example: Execute Bundle Flow
```bash
curl -X POST http://127.0.0.1:8000/bundle/execute-flow \
  -H "Content-Type: application/json" \
  -d '{"request_text": "iPhone 17 Pro bundle with case and charger under $1100"}'
```

#### Example: Execute Search Flow
```bash
curl -X POST http://127.0.0.1:8000/searchandising/execute-flow \
  -H "Content-Type: application/json" \
  -d '{"merchant_instruction": "Push high margin Nike running shoes with good color variety"}'
```

---

### 🧩 Granular Solver & Generator Endpoints

| Category | Endpoint | Description |
| :--- | :--- | :--- |
| **Bundle** | `POST /bundle/generate-rules` | Compiles prompt into structured JSON constraints. |
| **Bundle** | `POST /bundle/solve` | Executes pure Backtracking DFS on candidate products. |
| **Search** | `POST /searchandising/generate-params` | Generates QUBO $\alpha, \beta, \gamma$ weights via LLM. |
| **Search** | `POST /searchandising/optimize` | Runs Simulated Annealing on a candidate product list. |
| **Mock** | `GET /mock-constructor/search/{query}` | Synthesizes a realistic mock Constructor.io search response. |

---

## 🧪 Running Tests

Run the full mathematical solver and integration test suite:

```bash
PYTHONPATH=src pytest tests/
```

---

## 💻 Interactive Frontend Playground

Pair this backend with the companion Next.js frontend playground (`ConstructorAgentToolsFrontend`) to explore live interactive diagrams, test API endpoints in real-time, and visualize bundle graph solutions.

---

## 📄 License
MIT License. Crafted for high-performance e-commerce agent workflows.
