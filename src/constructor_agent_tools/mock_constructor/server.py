from fastapi import APIRouter, Request, HTTPException
from .llm_engine import llm_engine

router = APIRouter(prefix="/mock-constructor")

@router.get("/search/natural_language/{query}")
async def search_natural_language(query: str, request: Request):
    try:
        params = dict(request.query_params)
        params["query"] = query
        response = await llm_engine.process_request("Natural Language Search", params)
        return response
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Mock LLM Engine error: {str(e)}")

@router.get("/search/{query}")
async def search(query: str, request: Request):
    try:
        params = dict(request.query_params)
        params["query"] = query
        response = await llm_engine.process_request("Search", params)
        return response
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Mock LLM Engine error: {str(e)}")

@router.get("/browse/{filter_name}/{filter_value}")
async def browse(filter_name: str, filter_value: str, request: Request):
    try:
        params = dict(request.query_params)
        params["filter_name"] = filter_name
        params["filter_value"] = filter_value
        response = await llm_engine.process_request("Browse", params)
        return response
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Mock LLM Engine error: {str(e)}")

@router.get("/recommendations/v1/pods/{pod_id}")
async def get_recommendations(pod_id: str, request: Request):
    try:
        params = dict(request.query_params)
        params["pod_id"] = pod_id
        response = await llm_engine.process_request("Recommendations", params)
        return response
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Mock LLM Engine error: {str(e)}")

