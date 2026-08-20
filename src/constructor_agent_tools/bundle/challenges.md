# Technical Challenges & Architecture: E-Commerce Bundle Solvers

This document summarizes the architectural challenges of building real-time, logical product bundling solvers at scale (e.g., millions of SKUs) and details the strategies to address them.

---

## 1. The Scaling Problem: $O(N^2)$ Combinatorial Explosion
When creating a bundle containing $K$ slots from a catalog of $N$ products, a brute-force search space evaluates $N^K$ configurations.
*   **The Issue**: Evaluating product-to-product compatibility relationships manually is impossible at scale (millions of SKUs create trillions of possible pairs).
*   **Real-time Constraint**: E-commerce APIs must respond in under 100-300ms, making real-time search across huge candidate spaces unfeasible.

### Solutions
*   **Attribute-Based Compatibility (Graph Partitions)**: We define compatibility via matching attributes (e.g. matching `color` or `brand` facets) instead of product pairs. A single constraint rule scales dynamically to any number of SKUs.
*   **Graph Pruning & MRV Heuristics**: Before running the backtracking search, we prune the search space by category, exclusions, and rating requirements. Sorting slots by the Minimum Remaining Values (MRV) heuristic ensures we traverse the smallest branches first.

---

## 2. Two Notions of Compatibility: Soft vs. Hard

*   **Soft/Stylistic Compatibility**: Fuzzy, vibes-based relationships (e.g., "this shirt looks nice with these pants"). Similarity scores and vector embeddings are the correct tools for this.
*   **Hard/Logical Compatibility**: Strict physical or technical interoperability (e.g., "this phone has a USB-C port" $\leftrightarrow$ "this charger has a USB-C plug"). There is no partial credit; a mismatch represents a broken bundle.

---

## 3. Extending the Schema for Cross-Slot Key Mapping
To enforce hard logical constraints, the solver must compare different facet keys across different slots—for example, comparing a phone's `port_type` with a charger's `connector_type`.

We configure the schema to support this cross-slot key-to-key comparison:
```python
class CompatibilityConstraint(BaseModel):
    rule_name: str
    slot_a: str
    facet_a: str
    slot_b: str
    facet_b: str
    match_type: str = "exact"
```

During graph traversal, the solver reads the chosen product in `slot_a`, extracts the value of `facet_a`, and uses it to prune the candidates in `slot_b` before evaluating any pricing or similarity metrics.

---

## 4. The Real-Time Extraction Bottleneck
Most catalog data contains messy free-text descriptions (e.g., *"Features a reversible Type-C input for fast charging"*). Evaluating these descriptions using real-time LLM calls at request time will immediately cause API timeouts.

### The Decoupled Architecture
We split attribute extraction from query-time matching:
1.  **Offline (Background)**: An LLM agent processes unstructured text once at catalog ingestion time, extracting canonical metadata into structured facet key-values.
2.  **Online (Real-Time)**: The solver tool performs fast, deterministic comparisons on the pre-extracted facets.

```mermaid
flowchart TD
    subgraph Offline Pipeline (Background)
        A[Messy Raw Catalog] -->|LLM Agent extracts attributes| B[Structured Catalog with clean Facets]
    end
    subgraph Online Engine (Real-Time)
        C[User Request] -->|Bundle Solver Tool| D[Fast Graph Execution]
        B --> D
    end
```

---

## 5. Scaling Vector Search to Compatibility Problems
Can vector search alone solve logical compatibility? **Only partially.**

*   **Vector search is excellent for candidate generation (Recall)**: It narrows down 1,000,000 SKUs to the top 50 semantically matching accessories.
*   **Vector search fails at logical validation (Precision)**: Cosine similarity measures topical closeness, not logical truth. A Lightning cable described with generalized "fast charging" language can easily rank highly for a USB-C phone search.
*   **The Hybrid Approach**: Use vector/hybrid search to filter the candidate pool down to a small set (e.g., 50 products), then apply the strict structured facet matching filter to guarantee compatibility.

---

## 6. Cost Controls for Large-Scale SKU Extraction
Running LLM processing over millions of SKUs is financially unviable. We implement a cascade of cheaper controls:

1.  **80/20 Rule**: LLMs only process high-traffic ("head") products, and only within categories where hard compatibility is relevant (e.g., Electronics, Tools, Auto Parts—ignoring Apparel or Home Decor).
2.  **Regex/Small Models First**: High-performance local pattern matchers scan descriptions for connector standards (e.g. `USB-C|Type-C|Lightning`) at zero cost before escalating to hosted LLMs.
3.  **On-Demand Caching**: Parse product facets only when they are first requested in a live bundle flow, caching them permanently.
4.  **Keyword Search Pruning**: Query search indexes for key attributes first to narrow down the target group before running verification checks.

---

## 7. Scaling the Normalization Layer
Maintaining a global lookup dictionary of synonyms across all product types does not scale. Instead, we use three techniques:

1.  **Enum Constraints at Extraction**: Force the LLM to output predefined canonical values (e.g., forcing `"USB-C"` instead of allowing `"Type-C"` or `"USB Type C"`).
2.  **Clustering Residual Values**: Take unmapped residual strings, embed them, and cluster them so a human or cheap LLM review can merge synonyms in batches.
3.  **Category-Scoped Registry**: Scope schemas per category pair (e.g., Phones $\leftrightarrow$ Chargers) rather than globally. Each leaf category pair maintains its own independent, highly bounded compatibility schema.
