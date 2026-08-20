import json
import logging
from typing import List, Optional
from pydantic import BaseModel
from google import genai
from google.genai import types

from constructor_agent_tools.settings import settings
from constructor_agent_tools.bundle.schemas.schemas import BundleRule, BundleSlot

class BundleRuleList(BaseModel):
    rules: List[BundleRule]

class BundleSlotList(BaseModel):
    slots: List[BundleSlot]

logger = logging.getLogger("constructor_agent_tools.bundle.generator")

class BundleRuleGenerator:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.GEMINI_MODEL
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

    def _get_slots_system_prompt(self) -> str:
        return """You are a merchandising expert assistant for Constructor.io.
Your job is to translate natural language merchandising requests into logical slots for a product bundle.
For example, if the user asks for a "winter outfit bundle", you might generate slots like "Beanie", "Winter Coat", "Gloves".
Generate 2 to 5 highly relevant slots for the requested bundle. Provide a slot_id (e.g., "slot_1") and a display_name (e.g., "Winter Coat").
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
                response_schema=BundleRuleList,
                temperature=0.0,
            )
        )

        try:
            parsed = BundleRuleList.model_validate_json(response.text)
            return parsed.rules
        except Exception as e:
            logger.error(f"Failed to generate structured rules: {e}")
            logger.debug(f"Raw Response: {response.text}")
            raise

    async def generate_slots(self, request_text: str) -> List[BundleSlot]:
        user_prompt = f"Merchandiser Request: \"{request_text}\""

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=self._get_slots_system_prompt(),
                response_mime_type="application/json",
                response_schema=BundleSlotList,
                temperature=0.0,
            )
        )

        try:
            parsed = BundleSlotList.model_validate_json(response.text)
            return parsed.slots
        except Exception as e:
            logger.error(f"Failed to generate bundle slots: {e}")
            logger.debug(f"Raw Response: {response.text}")
            raise
