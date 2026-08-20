import json
import logging
from typing import Optional
from google import genai
from google.genai import types

from constructor_agent_tools.settings import settings
from pydantic import BaseModel
from constructor_agent_tools.searchandising.schemas import MerchandisingParams

class SearchQueryExtraction(BaseModel):
    query: str

logger = logging.getLogger("constructor_agent_tools.searchandising.generator")

SYSTEM_PROMPT = """You are a search merchandising optimization expert.

Given a merchandiser's natural language instruction about how to rank search results, generate the optimal weight parameters for a QUBO (Quadratic Unconstrained Binary Optimization) ranking model.

The model has three weight parameters:
1. alpha (relevance weight): Controls how much the search relevance score matters. Higher values mean products matching the query rank higher.
2. beta (business boost weight): Controls how much business metrics (profit margin, inventory level, sponsorship) matter. Higher values favor profitable or promoted products.
3. gamma (diversity penalty weight): Controls how much to penalize showing similar products next to each other. Higher values force a more diverse result set.

Additionally, specify num_slots: the number of products to select for the final page.

Default values are alpha=1.0, beta=0.5, gamma=0.3, num_slots=10.

Guidelines for interpreting merchant intent:
- "maximize revenue/profit" → increase beta significantly (e.g., 1.5-2.0)
- "show diverse results" / "variety" → increase gamma significantly (e.g., 0.8-1.5)
- "most relevant first" → increase alpha, decrease beta (e.g., alpha=2.0, beta=0.2)
- "promote sponsored items" → increase beta (e.g., 1.0-1.5)
- "show more results" → increase num_slots
- Balanced instructions → keep values moderate
"""


class MerchandisingParamsGenerator:
    """LLM-powered generator that translates merchant intent into QUBO weight parameters."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.GEMINI_MODEL
        self._client = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    async def generate_params(self, merchant_instruction: str) -> MerchandisingParams:
        """
        Translates a merchandiser's natural language instruction into
        structured QUBO weight parameters.
        """
        user_prompt = f'Merchandiser Instruction: "{merchant_instruction}"'

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=MerchandisingParams,
                temperature=0.1,
            ),
        )

        try:
            return MerchandisingParams.model_validate_json(response.text)
        except Exception as e:
            logger.error(f"Failed to parse merchandising params: {e}")
            logger.debug(f"Raw response: {response.text}")
            raise

    async def extract_base_query(self, merchant_instruction: str) -> str:
        """Extracts the base search query from a merchandising instruction."""
        user_prompt = f'Merchandiser Instruction: "{merchant_instruction}"'
        system_instruction = "Extract the core search term from the merchandiser's instruction. For example, 'Boost high margin laptops' -> 'laptops'. If no specific term is implied, return an empty string."
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=SearchQueryExtraction,
                temperature=0.0,
            ),
        )

        try:
            res = SearchQueryExtraction.model_validate_json(response.text)
            return res.query
        except Exception as e:
            logger.error(f"Failed to extract base query: {e}")
            logger.debug(f"Raw response: {response.text}")
            return ""
