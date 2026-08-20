from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import random

from constructor_agent_tools.mock_constructor.llm_engine import llm_engine

from constructor_agent_tools.searchandising.schemas import (
    SearchProduct,
    MerchandisingParams,
    OptimizeRankingRequest,
    OptimizeRankingResponse,
)
from constructor_agent_tools.searchandising.generator import MerchandisingParamsGenerator
from constructor_agent_tools.searchandising.qubo_solver import QUBOSolver

router = APIRouter(prefix="/searchandising")
params_generator = MerchandisingParamsGenerator()
qubo_solver = QUBOSolver()


def _safe_float(val, default: float) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = re.sub(r"[^\d.-]", "", str(val))
    try:
        return float(cleaned) if cleaned else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default: int) -> int:
    if val is None:
        return default
    if isinstance(val, int):
        return val
    cleaned = re.sub(r"[^\d-]", "", str(val))
    try:
        return int(float(cleaned)) if cleaned else default
    except (ValueError, TypeError):
        return default


class GenerateParamsRequest(BaseModel):
    merchant_instruction: str


@router.post("/generate-params", response_model=MerchandisingParams)
async def generate_params(req: GenerateParamsRequest):
    """Translate a merchandiser's natural language instruction into QUBO weight parameters."""
    try:
        params = await params_generator.generate_params(req.merchant_instruction)
        return params
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to generate params: {str(e)}")

class SearchFlowRequest(BaseModel):
    merchant_instruction: str
    products: Optional[List[SearchProduct]] = Field(default_factory=list)

class SearchFlowResponse(BaseModel):
    generated_params: MerchandisingParams
    optimization_result: OptimizeRankingResponse

@router.post("/execute-flow", response_model=SearchFlowResponse)
async def execute_search_flow(req: SearchFlowRequest):
    """Executes the end-to-end search flow: LLM Params Generation -> QUBO Optimization"""
    try:
        # Agentic Fetch: Products
        products = req.products
        if not products:
            base_query = await params_generator.extract_base_query(req.merchant_instruction)
            if not base_query:
                base_query = "all products"
            
            mock_res = await llm_engine.process_request("Search", {"query": base_query})
            for res in mock_res.get("response", {}).get("results", []):
                item = res.get("data", {})
                facets = item.get("facets", {})
                
                # Parse realistic defaults or facets safely
                price = _safe_float(facets.get("price"), random.uniform(15.0, 150.0))
                relevance_score = _safe_float(facets.get("relevance"), random.uniform(0.5, 1.0))
                margin = _safe_float(facets.get("margin"), random.uniform(0.05, 0.40))
                inventory = _safe_int(facets.get("inventory"), random.randint(0, 500))
                is_sponsored = random.random() < 0.2  # 20% chance to be sponsored
                
                products.append(SearchProduct(
                    id=item.get("id"),
                    title=res.get("value"),
                    category=facets.get("category", "General"),
                    price=price,
                    relevance_score=relevance_score,
                    margin=margin,
                    inventory=inventory,
                    is_sponsored=is_sponsored,
                    facets=facets
                ))

        # Step 1: Generate params
        params = await params_generator.generate_params(req.merchant_instruction)
        
        # Step 2: Optimize Ranking
        solver = QUBOSolver()
        result = solver.solve(products, params)
        
        return SearchFlowResponse(
            generated_params=params,
            optimization_result=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute search flow: {str(e)}")


@router.post("/optimize", response_model=OptimizeRankingResponse)
async def optimize_ranking(req: OptimizeRankingRequest):
    """Run the QUBO optimizer on the given products with the specified parameters."""
    try:
        solver = QUBOSolver()
        return solver.solve(req.products, req.params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize ranking: {str(e)}")
