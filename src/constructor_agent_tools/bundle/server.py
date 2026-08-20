from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import random

from constructor_agent_tools.bundle.schemas import (
    BundleRule,
    BundleSlot,
    BundleProduct,
    BundleSolutionRequest,
    BundleSolutionResponse,
)
from constructor_agent_tools.bundle.generator import BundleRuleGenerator
from constructor_agent_tools.bundle.solver import BundleSolver
from constructor_agent_tools.mock_constructor.llm_engine import llm_engine

router = APIRouter(prefix="/bundle")
rule_generator = BundleRuleGenerator()

class GenerateRulesRequest(BaseModel):
    request_text: str
    slots: List[BundleSlot]

@router.post("/generate-rules", response_model=List[BundleRule])
async def generate_rules(req: GenerateRulesRequest):
    try:
        rules = await rule_generator.generate_rules(req.request_text, req.slots)
        return rules
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to generate rules: {str(e)}")

import re

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

class BundleFlowRequest(BaseModel):
    request_text: str
    slots: Optional[List[BundleSlot]] = Field(default_factory=list)
    products: Optional[List[BundleProduct]] = Field(default_factory=list)

class BundleFlowResponse(BaseModel):
    generated_rules: List[BundleRule]
    solution: BundleSolutionResponse

@router.post("/execute-flow", response_model=BundleFlowResponse)
async def execute_bundle_flow(req: BundleFlowRequest):
    """Executes the end-to-end bundle flow: LLM Rule Generation -> Mathematical Solving"""
    try:
        # Agentic Generation: Slots
        slots = req.slots
        if not slots:
            slots = await rule_generator.generate_slots(req.request_text)

        # Agentic Fetch: Products
        products = req.products
        if not products:
            for slot in slots:
                mock_res = await llm_engine.process_request("Search", {"query": slot.display_name})
                for res in mock_res.get("response", {}).get("results", []):
                    item = res.get("data", {})
                    facets = item.get("facets", {})
                    facets["slot_id"] = slot.slot_id
                    # Parse realistic defaults or facets safely
                    price = _safe_float(facets.get("price"), random.uniform(15.0, 150.0))
                    rating = _safe_float(facets.get("rating"), random.uniform(3.5, 5.0))
                    
                    products.append(BundleProduct(
                        id=item.get("id", f"p_{random.randint(1000, 9999)}"),
                        name=res.get("value", slot.display_name),
                        category=slot.display_name,
                        price=price,
                        rating=rating,
                        facets=facets
                    ))
                    
        # Step 1: Generate Rules
        rules = await rule_generator.generate_rules(req.request_text, slots)
        
        # Step 2: Solve Bundle
        solver = BundleSolver(slots, products, rules)
        solution = solver.solve()
        
        return BundleFlowResponse(
            generated_rules=rules,
            solution=solution
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute bundle flow: {str(e)}")

@router.post("/solve", response_model=BundleSolutionResponse)
async def solve_bundle(req: BundleSolutionRequest):
    try:
        solver = BundleSolver(req.slots, req.products, req.rules)
        return solver.solve()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to solve bundle: {str(e)}")
