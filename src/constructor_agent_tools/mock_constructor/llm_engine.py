import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from constructor_agent_tools.settings import settings

logger = logging.getLogger("constructor_agent_tools.mock_constructor")

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

# We define the LLM response schema without client-side variables to let the LLM focus on the response payload
class ConstructorAPIResponse(BaseModel):
    response: ConstructorResponsePayload

class LLMMockEngine:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self._client = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            # Lazy initialization to prevent startup crashes when API key is missing
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                logger.warning("GEMINI_API_KEY is not set. Mock LLM requests will fail.")
            self._client = genai.Client(api_key=api_key)
        return self._client

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

    async def process_request(self, endpoint_type: str, request_params: Dict[str, Any]) -> Dict[str, Any]:
        system_instruction = self._get_system_prompt(endpoint_type)
        user_prompt = f"Process this {endpoint_type} API request: {json.dumps(request_params)}"
        
        # Use client.aio for asynchronous, non-blocking calls
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ConstructorAPIResponse,
                temperature=0.0,
            ),
        )
        
        try:
            parsed_data = ConstructorAPIResponse.model_validate_json(response.text)
            
            # Combine the parsed payload with server-side variables (echoing the request, and generating a UUID)
            return {
                "request": request_params,
                "response": parsed_data.response.model_dump(),
                "result_id": str(uuid.uuid4())
            }
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Raw Response: {response.text}")
            raise

llm_engine = LLMMockEngine()
