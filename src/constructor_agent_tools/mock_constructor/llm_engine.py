import os
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from constructor_agent_tools.settings import settings

# Define Pydantic models for structured output matching Constructor's Schema
class ConstructorGroup(BaseModel):
    group_id: str
    display_name: str

class ConstructorItemData(BaseModel):
    id: str
    image_url: Optional[str] = None
    url: Optional[str] = None
    facets: Dict[str, Any] = Field(default_factory=dict)
    groups: List[ConstructorGroup] = Field(default_factory=list)

class ConstructorResult(BaseModel):
    value: str
    data: ConstructorItemData
    matched_terms: List[str] = Field(default_factory=list)

class ConstructorFacetOption(BaseModel):
    value: str
    display_name: str
    count: int

class ConstructorFacet(BaseModel):
    name: str
    display_name: str
    type: str = "multiple"
    options: List[ConstructorFacetOption] = Field(default_factory=list)

class ConstructorResponsePayload(BaseModel):
    total_num_results: int
    results: List[ConstructorResult]
    facets: List[ConstructorFacet] = Field(default_factory=list)

class ConstructorAPIResponse(BaseModel):
    request: Dict[str, Any]
    response: ConstructorResponsePayload
    result_id: str

class LLMMockEngine:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.client = genai.Client(api_key=settings.gemini_api_key)
        
    def _get_system_prompt(self, endpoint_type: str) -> str:
        return f"""You are a mock implementation of the Constructor.io {endpoint_type} API.
You must act as a dynamic, intelligent search and discovery engine.

You DO NOT have a predefined product catalog. Instead, you must **synthesize a highly realistic, plausible set of products on demand** that perfectly matches the user's requested query, browse filters, and pre-filter expressions.

INSTRUCTIONS:
1. Analyze the request payload to understand what the user is looking for (e.g., query="red shoes" or filter_name="category" filter_value="laptops").
2. Invent 5-15 highly realistic products that match this intent. Give them realistic IDs, names, image URLs, prices, and brands.
3. If pre_filter_expressions or specific facet filters are provided, ensure your synthesized products strictly adhere to those constraints.
4. Dynamically generate realistic facet options and groups (categories) based on the synthetic products you just invented.
5. Sort results by relevance (attractiveness) unless otherwise specified in the request.
6. Return ONLY valid JSON matching the exact schema required.
"""

    def process_request(self, endpoint_type: str, request_params: Dict[str, Any]) -> ConstructorAPIResponse:
        system_instruction = self._get_system_prompt(endpoint_type)
        user_prompt = f"Process this {endpoint_type} API request: {json.dumps(request_params)}"
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ConstructorAPIResponse,
                temperature=0.0 # Deterministic structured output
            ),
        )
        
        # Parse the JSON string returned by the model back into our Pydantic model
        # (Though `response.parsed` might be available depending on the exact SDK version, manual parsing is safer)
        try:
            return ConstructorAPIResponse.model_validate_json(response.text)
        except Exception as e:
            # Fallback for debugging if LLM fails
            print(f"Error parsing LLM response: {e}")
            print(f"Raw Response: {response.text}")
            raise

llm_engine = LLMMockEngine()
