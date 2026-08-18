from typing import Any, Optional
from fastapi import APIRouter, Request
from .llm_engine import llm_engine

router = APIRouter(prefix="/mock-constructor")

@router.get("/search/{query}")
async def search(query: str, request: Request):
    params = dict(request.query_params)
    params["query"] = query
    response = llm_engine.process_request("Search", params)
    return response.model_dump()

@router.get("/search/natural_language/{query}")
async def search_natural_language(query: str, request: Request):
    params = dict(request.query_params)
    params["query"] = query
    response = llm_engine.process_request("Natural Language Search", params)
    return response.model_dump()

@router.get("/browse/{filter_name}/{filter_value}")
async def browse(filter_name: str, filter_value: str, request: Request):
    params = dict(request.query_params)
    params["filter_name"] = filter_name
    params["filter_value"] = filter_value
    response = llm_engine.process_request("Browse", params)
    return response.model_dump()

@router.get("/recommendations/v1/pods/{pod_id}")
async def get_recommendations(pod_id: str, request: Request):
    params = dict(request.query_params)
    params["pod_id"] = pod_id
    response = llm_engine.process_request("Recommendations", params)
    return response.model_dump()
