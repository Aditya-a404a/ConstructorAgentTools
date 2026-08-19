import json
import logging
from typing import List, Optional
from google import genai
from google.genai import types

from constructor_agent_tools.settings import settings
from constructor_agent_tools.bundle.schemas.schemas import BundleRule, BundleSlot

logger = logging.getLogger("constructor_agent_tools.bundle.generator")

class BundleRuleGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self._client = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    def _get_system_prompt(self) -> str:
        return """You are a merchandising expert assistant for Constructor.io.
Your job is to translate natural language merchandising requests into structured rules for generating product bundles.

You will be given:
1. A natural language request from a merchandiser.
2. A list of target slots in the bundle.

You must generate a list of structured `BundleRule` objects matching the specified schema.
Make sure you use the exact slot IDs provided in the request slots.
For compatibility constraints, identify which facets (e.g., "color", "brand", "material") should match between slots.
For price constraints, make sure the rules restrict maximum slot pricing or overall pricing.
For rating constraints, apply reasonable minimum rating requirements based on instructions.
"""

    async def generate_rules(self, request_text: str, slots: List[BundleSlot]) -> List[BundleRule]:
        # Formulate instructions for LLM
        slots_json = json.dumps([s.model_dump() for s in slots])
        user_prompt = f"Merchandiser Request: \"{request_text}\"\nAvailable Slots in Bundle: {slots_json}"

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=self._get_system_prompt(),
                response_mime_type="application/json",
                response_schema=List[BundleRule],
                temperature=0.0,
            )
        )

        try:
            # Parse response back into the Pydantic model list
            # We wrap the JSON string with a parsing utility
            import pydantic
            adapter = pydantic.TypeAdapter(List[BundleRule])
            return adapter.validate_json(response.text)
        except Exception as e:
            logger.error(f"Failed to generate structured rules: {e}")
            logger.debug(f"Raw Response: {response.text}")
            raise
