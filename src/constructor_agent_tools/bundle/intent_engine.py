import json
import logging
from typing import List
from google import genai
from google.genai import types

from constructor_agent_tools.settings import settings
from constructor_agent_tools.bundle.dynamic_schemas import (
    CatalogProduct,
    IntentResult,
    SlotRule,
)

logger = logging.getLogger("constructor_agent_tools.bundle.intent_engine")

SYSTEM_PROMPT = """You are an expert e-commerce merchandising assistant.

Given:
1. An anchor product the user is currently viewing.
2. The user's natural language request describing what kind of bundle they want.

Your job:
1. Determine if the user's intent requires HARD functional compatibility (e.g., electronics accessories that must physically work together) or SOFT stylistic compatibility (e.g., fashion items that look good together). Output "hard" or "soft".
2. Generate a list of bundle slots. Each slot represents a product category/type to include in the bundle.
3. For each slot, write a VERBOSE, DETAILED compatibility rule description. This description will be used as a search query to find matching products via semantic search. Be as specific as possible about:
   - Required specifications, standards, or connector types (for hard compatibility)
   - Style, color palette, material, or aesthetic (for soft compatibility)
   - Any constraints derived from the anchor product's attributes
4. Optionally suggest a category_hint for each slot to narrow down search results.
5. If the user mentions a budget, set max_total_price accordingly.

IMPORTANT: The rule_description for each slot must be written as if you are describing the IDEAL product for that slot, not as a constraint definition. Write it the way a product listing would describe the item.
"""


class IntentEngine:
    """LLM Call 1: Analyzes user intent and generates verbose slot rules."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self._client = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    async def analyze_intent(
        self, anchor_product: CatalogProduct, user_request: str
    ) -> IntentResult:
        """
        Takes the anchor product and user request, returns structured
        IntentResult with slot rules for vector search.
        """
        anchor_info = json.dumps({
            "id": anchor_product.id,
            "title": anchor_product.title,
            "description": anchor_product.description,
            "category": anchor_product.category,
            "price": anchor_product.price,
            "facets": anchor_product.facets,
        })

        user_prompt = (
            f"Anchor Product:\n{anchor_info}\n\n"
            f"User Request: \"{user_request}\""
        )

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=IntentResult,
                temperature=0.2,
            ),
        )

        try:
            return IntentResult.model_validate_json(response.text)
        except Exception as e:
            logger.error(f"Failed to parse intent result: {e}")
            logger.debug(f"Raw response: {response.text}")
            raise
