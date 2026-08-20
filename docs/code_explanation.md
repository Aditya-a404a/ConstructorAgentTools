# Constructor Agent Tools — Complete Code Explanation

This document provides a line-by-line explanation of every file in the project.

---

# Part 1: Application Foundation

---

## 1. `settings.py` — Configuration Management

```python
from typing import Optional
from pydantic_settings import BaseSettings
```
- `Optional` allows fields to have `None` as a valid value.
- `BaseSettings` from `pydantic_settings` auto-loads values from environment variables and `.env` files.

```python
class Settings(BaseSettings):
    APP_NAME: str = "Constructor Agent Tools"  # Display name for the FastAPI docs
    VERSION: str = "0.1.0"                     # Application version
    DEBUG: bool = True                         # Toggle debug mode
```
Each field is typed. The default values are used when no env var is set.

```python
    CONSTRUCTOR_API_URL: str = "http://localhost:8001/mock-constructor"
    CONSTRUCTOR_API_KEY: Optional[str] = None
```
- `CONSTRUCTOR_API_URL`: Points to our own mock server (port 8001 matches `__main__.py`).
- `CONSTRUCTOR_API_KEY`: Placeholder for real Constructor.io API keys (unused in mock mode).

```python
    GEMINI_API_KEY: Optional[str] = None
```
The API key for Google Gemini. Used by the mock engine, rule generator, intent engine, and embedding engine.

```python
    BUNDLE_SOLVER_TIMEOUT_SECONDS: float = 10.0
```
Maximum time (in seconds) the bundle solver is allowed to search before returning the best results found so far.

```python
    model_config = {
        "env_file": ".env",           # Reads from a .env file in the project root
        "env_file_encoding": "utf-8",
        "extra": "ignore"             # Ignores unknown env vars instead of crashing
    }
```
Pydantic v2 configuration. Replaces the old `class Config:` pattern.

```python
settings = Settings()
```
Module-level singleton. Imported everywhere as `from constructor_agent_tools.settings import settings`.

---

## 2. `__main__.py` — CLI Entry Point

```python
import uvicorn

def main():
    uvicorn.run(
        "constructor_agent_tools.main:app",  # Import path to the FastAPI app object
        host="127.0.0.1",                    # Listen on localhost only
        port=8001,                           # The port number
        reload=True,                         # Auto-reload on code changes (dev mode)
    )

if __name__ == "__main__":
    main()
```
- Running `python -m constructor_agent_tools` triggers this file.
- `uvicorn.run(...)` starts the ASGI server pointing at the FastAPI `app` object in `main.py`.

---

## 3. `main.py` — FastAPI Application Setup

```python
from fastapi import FastAPI
from constructor_agent_tools.settings import settings
from constructor_agent_tools.mock_constructor.server import router as mock_router
```
Imports the settings singleton and the mock constructor's API router.

```python
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Advanced autonomous agent systems built on Constructor.io APIs",
)
```
Creates the FastAPI application. These fields populate the auto-generated `/docs` Swagger page.

```python
app.include_router(mock_router)
```
Mounts all the mock constructor endpoints under the `/mock-constructor` prefix.

```python
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": settings.VERSION}
```
A simple health check endpoint. Returns `200 OK` with the app version.

---

# Part 2: Mock Constructor API

---

## 4. `mock_constructor/llm_engine.py` — LLM-Powered Mock Search Engine

### Pydantic Response Schemas (Lines 13–48)

```python
class ConstructorGroup(BaseModel):
    group_id: str          # e.g., "cat_shoes"
    display_name: str      # e.g., "Shoes"
```
Represents a product category/group in Constructor.io's response format.

```python
class ConstructorItemData(BaseModel):
    id: str                                # Product ID
    image_url: Optional[str] = None        # Product image
    url: Optional[str] = None              # Product page link
    facets: Dict[str, Any] = ...           # Filterable attributes (brand, color, size...)
    groups: List[ConstructorGroup] = ...   # Categories this product belongs to
```
The nested data object for each search result.

```python
class ConstructorResult(BaseModel):
    value: str                             # Display name of the product
    data: ConstructorItemData              # Nested product data
    matched_terms: List[str] = ...         # Which search terms matched
```
A single search result.

```python
class ConstructorFacetOption(BaseModel):
    value: str             # e.g., "Nike"
    display_name: str      # e.g., "Nike"
    count: int             # Number of products matching this facet
```

```python
class ConstructorFacet(BaseModel):
    name: str              # e.g., "brand"
    display_name: str      # e.g., "Brand"
    type: str = "multiple" # Facet type (multiple = checkboxes)
    options: List[ConstructorFacetOption] = ...
```
Facets are the filter sidebar in search results (Brand, Color, Size, etc.).

```python
class ConstructorResponsePayload(BaseModel):
    total_num_results: int
    results: List[ConstructorResult]
    facets: List[ConstructorFacet] = ...
```
The main response body.

```python
class ConstructorAPIResponse(BaseModel):
    response: ConstructorResponsePayload
```
The LLM only generates the `response` payload. We add `request` and `result_id` server-side.

### The Engine Class (Lines 50–110)

```python
class LLMMockEngine:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self._client = None  # Lazy — not created until first use
```

```python
    @property
    def client(self) -> genai.Client:
        if self._client is None:
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                logger.warning("GEMINI_API_KEY is not set.")
            self._client = genai.Client(api_key=api_key)
        return self._client
```
**Lazy initialization**: The Gemini client is only created when the first request comes in. This prevents import-time crashes if the API key is missing.

```python
    async def process_request(self, endpoint_type: str, request_params: Dict[str, Any]) -> Dict[str, Any]:
```
- `async` so it doesn't block the FastAPI event loop.
- `endpoint_type` is "Search", "Browse", etc. — used in the system prompt.

```python
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",    # Force JSON output
                response_schema=ConstructorAPIResponse,   # Force schema compliance
                temperature=0.0,                          # Deterministic
            ),
        )
```
- `client.aio` — async, non-blocking Gemini call.
- `response_schema` — Gemini's structured output mode forces the LLM to return valid JSON matching our Pydantic model.

```python
        try:
            parsed_data = ConstructorAPIResponse.model_validate_json(response.text)
            return {
                "request": request_params,              # Echo original request (server-side)
                "response": parsed_data.response.model_dump(),
                "result_id": str(uuid.uuid4())          # Server-generated UUID
            }
```
- We parse the LLM's JSON into our Pydantic model for validation.
- `request` and `result_id` are populated server-side (not by the LLM).

```python
llm_engine = LLMMockEngine()
```
Module-level singleton. Safe because of lazy initialization — no API calls happen here.

---

## 5. `mock_constructor/server.py` — FastAPI Routes

```python
router = APIRouter(prefix="/mock-constructor")
```
All routes in this file are prefixed with `/mock-constructor`.

```python
@router.get("/search/natural_language/{query}")   # REGISTERED FIRST — more specific
async def search_natural_language(query: str, request: Request):
```
**Route ordering matters**: This route is registered before `/search/{query}` to prevent FastAPI from matching `natural_language` as a query parameter.

```python
    try:
        params = dict(request.query_params)  # Extract URL query params (?page=1&limit=10)
        params["query"] = query              # Add the path parameter
        response = await llm_engine.process_request("Natural Language Search", params)
        return response                      # FastAPI auto-serializes dicts to JSON
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Mock LLM Engine error: {str(e)}")
```
- `await` — non-blocking call to the async LLM engine.
- `502 Bad Gateway` — signals that the upstream mock engine failed.

The remaining routes (`/search/{query}`, `/browse/{filter_name}/{filter_value}`, `/recommendations/v1/pods/{pod_id}`) follow the exact same pattern.

---

# Part 3: Bundle Solver (Rule-Based, V1)

---

## 6. `bundle/schemas/schemas.py` — Rule-Based Bundle Schemas

```python
class BundleProduct(BaseModel):
    id: str
    name: str
    category: str         # e.g., "tops", "bottoms"
    price: float
    rating: float = 5.0
    facets: Dict[str, Any] = ...  # {"color": "red", "brand": "Nike"}
    score: float = 1.0    # Search relevance / popularity score
```
A product in the catalog with structured metadata.

```python
class CategoryConstraint(BaseModel):
    slot_id: str    # Which slot this constraint applies to (e.g., "top")
    category: str   # Required category (e.g., "tops")
```
Maps a slot to a product category.

```python
class MinRatingConstraint(BaseModel):
    slot_id: Optional[str] = None  # None = applies to ALL slots
    min_rating: float
```

```python
class PriceConstraint(BaseModel):
    max_total_price: Optional[float] = None           # Cap on the entire bundle
    max_slot_price: Dict[str, float] = ...            # Per-slot price caps
```

```python
class CompatibilityConstraint(BaseModel):
    facet_name: str               # e.g., "color"
    match_type: str = "exact"     # All products in the bundle must have the same value
```
Enforces that a specific facet (like "color") matches across all products in the bundle.

```python
class BundleRule(BaseModel):
    rule_name: str
    description: str
    category_constraints: List[CategoryConstraint] = ...
    min_rating_constraints: List[MinRatingConstraint] = ...
    price_constraints: Optional[PriceConstraint] = None
    compatibility_constraints: List[CompatibilityConstraint] = ...
    required_product_ids: List[str] = ...    # Must include these products
    excluded_product_ids: List[str] = ...    # Must NOT include these products
```
A rule is a container holding multiple constraint types.

```python
class BundleSolution(BaseModel):
    products: Dict[str, BundleProduct]  # slot_id -> chosen product
    total_price: float
    average_rating: float
    total_score: float
```
One valid bundle configuration.

---

## 7. `bundle/generator.py` — LLM Rule Generator

```python
async def generate_rules(self, request_text: str, slots: List[BundleSlot]) -> List[BundleRule]:
    slots_json = json.dumps([s.model_dump() for s in slots])
    user_prompt = f"Merchandiser Request: \"{request_text}\"\nAvailable Slots in Bundle: {slots_json}"
```
Serializes the slot definitions and user request into a prompt.

```python
    response = await self.client.aio.models.generate_content(
        ...
        config=types.GenerateContentConfig(
            response_schema=List[BundleRule],  # Forces LLM to output a list of rules
            temperature=0.0,                   # Deterministic
        )
    )
```

```python
    import pydantic
    adapter = pydantic.TypeAdapter(List[BundleRule])
    return adapter.validate_json(response.text)
```
- `TypeAdapter` is needed because `model_validate_json` only works on single models, not `List[Model]`.
- This parses the LLM's JSON array into a validated list of `BundleRule` objects.

---

## 8. `bundle/solver.py` — Multipartite Graph Constraint Solver

### Initialization (Lines 18–29)
```python
    self.category_to_products: Dict[str, List[BundleProduct]] = {}
    for p in products:
        self.category_to_products.setdefault(p.category, []).append(p)
```
Pre-indexes products by category for fast lookup.

### Candidate Filtering (Lines 31–89)
For each slot, the solver:
1. **Category filter**: Only keeps products matching the slot's required category.
2. **Exclusion filter**: Removes any product in `excluded_product_ids`.
3. **Rating filter**: Removes products below `min_rating`.
4. **Slot price filter**: Removes products exceeding per-slot price caps.

### MRV Heuristic (Line 96)
```python
sorted_slots = sorted(self.slots, key=lambda s: len(slot_candidates.get(s.slot_id, [])))
```
**Minimum Remaining Values**: Slots with fewer candidates are processed first. This causes the backtracking search to fail fast on constrained slots rather than wasting time exploring easy slots first.

### Backtracking DFS (Lines 98–152)
```python
def backtrack(slot_idx: int):
    if time.time() - start_time > self.timeout:
        return  # Timeout — stop searching
```
Every recursion checks the clock.

```python
    if slot_idx == len(sorted_slots):
        # All slots filled! Check bundle-wide constraints:
        # 1. Total price cap
        # 2. Compatibility (all products must share the same facet value)
        # 3. Required products must be present
        # If all pass → save as a valid solution
        return
```

```python
    for product in slot_candidates[sid]:
        if product.id in [p.id for p in current_solution.values()]:
            continue  # No duplicate products across slots
        current_solution[sid] = product
        backtrack(slot_idx + 1)       # Recurse to next slot
        del current_solution[sid]     # Undo (backtrack)
```
Standard DFS backtracking: try a product, recurse, undo, try the next.

---

# Part 4: Dynamic Intent + Vector Search Bundle Solver (V2)

---

## 9. `bundle/dynamic_schemas.py` — Vector Search Schemas

```python
class CatalogProduct(BaseModel):
    id: str
    title: str
    description: str = ""     # Free-text product description (used for embeddings)
    category: str = ""
    price: float = 0.0
    rating: float = 5.0
    image_url: Optional[str] = None
    facets: Dict[str, Any] = ...
```
Similar to `BundleProduct` but includes `title` and `description` (needed for embedding generation).

```python
class SlotRule(BaseModel):
    slot_id: str
    display_name: str
    rule_description: str           # Verbose NL description — becomes the vector search query
    category_hint: Optional[str]    # Optional category to narrow search scope
```
The key innovation: `rule_description` is written by the LLM in product-listing style so it lives in the same semantic space as real product descriptions.

```python
class IntentResult(BaseModel):
    intent_type: str = "soft"              # "soft" or "hard"
    slots: List[SlotRule]
    max_total_price: Optional[float]       # Budget constraint from user request
```
Output of LLM Call 1.

```python
class ScoredCandidate(BaseModel):
    product: CatalogProduct
    similarity_score: float    # Cosine similarity between rule and product embedding
```

```python
class DynamicBundleSolution(BaseModel):
    slot_products: Dict[str, CatalogProduct]   # slot_id -> product
    slot_scores: Dict[str, float]              # slot_id -> similarity score
    total_price: float
    aggregate_similarity: float                # Mean of all slot similarity scores
```

---

## 10. `bundle/intent_engine.py` — LLM Call 1

```python
SYSTEM_PROMPT = """You are an expert e-commerce merchandising assistant.
...
IMPORTANT: The rule_description for each slot must be written as if you are 
describing the IDEAL product for that slot, not as a constraint definition.
Write it the way a product listing would describe the item.
"""
```
This instruction is critical: by telling the LLM to write rule descriptions like product listings, we ensure the generated text is semantically close to actual product descriptions in the vector store, maximizing retrieval quality.

```python
async def analyze_intent(self, anchor_product: CatalogProduct, user_request: str) -> IntentResult:
    anchor_info = json.dumps({...})   # Serialize anchor product for the prompt
    user_prompt = f"Anchor Product:\n{anchor_info}\n\nUser Request: \"{user_request}\""
```

```python
    response = await self.client.aio.models.generate_content(
        ...
        config=types.GenerateContentConfig(
            response_schema=IntentResult,   # Structured output
            temperature=0.2,               # Slightly creative for diverse rule descriptions
        ),
    )
    return IntentResult.model_validate_json(response.text)
```

---

## 11. `bundle/embedding_engine.py` — Call 2 + Vector Store

### VectorStore Class (Lines 18–148)

```python
def __init__(self, store_dir: Optional[Path] = None):
    self.store_dir = store_dir or DEFAULT_STORE_DIR
    self._ids: List[str] = []                      # Product IDs, index-aligned with vectors
    self._vectors: Optional[np.ndarray] = None     # Shape: (N, D) — N products, D dimensions
    self._metadata: Dict[str, dict] = {}           # product_id -> {title, price, category, ...}
    self._loaded = False                           # Lazy-load flag
```

```python
def _ensure_loaded(self):
    if not self._loaded:
        self.load()           # Load from disk on first access
        self._loaded = True
```

#### Adding Vectors
```python
def add(self, product_id, vector, metadata=None):
    vec = np.array(vector, dtype=np.float32)
    if self._vectors is None:
        self._vectors = vec.reshape(1, -1)       # First vector: create the matrix
    else:
        self._vectors = np.vstack([...])         # Append row to existing matrix
    self._ids.append(product_id)
```

#### Search (Cosine Similarity)
```python
def search(self, query_vector, top_k=50, category_filter=None):
    query = np.array(query_vector, dtype=np.float32)

    # Cosine similarity = dot(a, b) / (||a|| * ||b||)
    norms = np.linalg.norm(self._vectors, axis=1)   # Norm of each stored vector
    query_norm = np.linalg.norm(query)               # Norm of query vector
    similarities = np.dot(self._vectors, query) / (safe_norms * safe_query_norm)
```
This computes cosine similarity between the query and ALL stored vectors in one vectorized NumPy operation.

```python
    if category_filter:
        mask = np.array([
            self._metadata.get(pid, {}).get("category", "") == category_filter
            for pid in self._ids
        ])
        similarities = np.where(mask, similarities, -1.0)  # Zero out non-matching categories
```
Category filtering: sets similarity to -1 for products outside the target category, so they never appear in results.

```python
    top_indices = np.argsort(similarities)[::-1][:top_k]  # Sort descending, take top K
```

#### Persistence
```python
def save(self):
    np.save(str(self.store_dir / "vectors.npy"), self._vectors)   # Binary NumPy format
    json.dump(self._ids, open(self.store_dir / "ids.json", "w"))  # Product IDs
    json.dump(self._metadata, open(...))                          # Product metadata

def load(self):
    self._vectors = np.load(str(vectors_path))
    self._ids = json.load(open(ids_path))
    self._metadata = json.load(open(metadata_path))
```
Three files on disk: `vectors.npy` (binary), `ids.json`, `metadata.json`.

### EmbeddingEngine Class (Lines 151–203)

```python
async def embed_texts(self, texts: List[str]) -> List[List[float]]:
    result = await self.client.aio.models.embed_content(
        model=self.model_name,     # "text-embedding-004"
        contents=texts,            # Batch of strings
    )
    return [e.values for e in result.embeddings]
```
**Call 2**: Sends all rule descriptions (or product texts) in a single batch API call. Returns a list of float vectors.

```python
async def embed_catalog(self, products: List[CatalogProduct], vector_store: VectorStore):
    texts = [f"{p.title}. {p.description}" for p in products]
    embeddings = await self.embed_texts(texts)
    vector_store.add_batch(product_ids, embeddings, metadata_list)
    vector_store.save()
```
Pre-processes the catalog: concatenates title + description, embeds them, stores in the vector store, and persists to disk.

---

## 12. `bundle/dynamic_solver.py` — The Orchestrator

### Candidate Retrieval (Lines 37–72)
```python
async def retrieve_candidates(self, intent: IntentResult) -> Dict[str, List[ScoredCandidate]]:
    # Batch embed all rule descriptions in a single API call (Call 2)
    rule_texts = [slot.rule_description for slot in intent.slots]
    rule_vectors = await self.embedding_engine.embed_texts(rule_texts)
```
Embeds ALL slot rules in one batch call.

```python
    for slot, rule_vec in zip(intent.slots, rule_vectors):
        results = self.vector_store.search(
            query_vector=rule_vec,
            top_k=self.top_k_per_slot,
            category_filter=slot.category_hint,
        )
```
For each slot, runs a vector search using the rule embedding as the query. Optionally filters by category.

```python
        for product_id, score, metadata in results:
            product = CatalogProduct(
                id=metadata.get("id", product_id),
                title=metadata.get("title", ""),
                ...
            )
            candidates.append(ScoredCandidate(product=product, similarity_score=score))
```
Reconstructs `CatalogProduct` objects from stored metadata.

### The Solver (Lines 74–141)

```python
def solve(self, intent, candidates_per_slot) -> List[DynamicBundleSolution]:
    current_products: Dict[str, CatalogProduct] = {}   # Current partial assignment
    current_scores: Dict[str, float] = {}              # Similarity scores
    used_ids: set = set()                              # Prevent same product in two slots
```

```python
    def backtrack(slot_idx: int):
        if time.time() - start_time > self.timeout:
            return                                     # Timeout guard

        if slot_idx == len(slots):
            total_price = sum(p.price for p in current_products.values())
            if max_price is not None and total_price > max_price:
                return                                 # Over budget — reject

            aggregate = sum(current_scores.values()) / len(current_scores)
            solutions.append(DynamicBundleSolution(...))
            return
```

```python
        for candidate in candidates:
            if candidate.product.id in used_ids:
                continue                               # Dedup check

            current_products[slot.slot_id] = candidate.product
            current_scores[slot.slot_id] = candidate.similarity_score
            used_ids.add(candidate.product.id)

            backtrack(slot_idx + 1)                    # Recurse

            del current_products[slot.slot_id]         # Undo
            del current_scores[slot.slot_id]
            used_ids.discard(candidate.product.id)
```

```python
    solutions.sort(key=lambda s: (-s.aggregate_similarity, s.total_price))
```
Best bundles first: highest similarity, lowest price.

### End-to-End (Lines 143–149)
```python
async def generate_bundle(self, intent: IntentResult) -> List[DynamicBundleSolution]:
    candidates = await self.retrieve_candidates(intent)  # Call 2 + vector search
    return self.solve(intent, candidates)                 # Graph solve (0 API calls)
```
The full pipeline in two lines.
