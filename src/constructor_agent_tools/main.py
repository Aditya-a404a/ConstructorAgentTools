from fastapi import FastAPI
from constructor_agent_tools.settings import settings
from constructor_agent_tools.mock_constructor.server import router as mock_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Advanced autonomous agent systems built on Constructor.io APIs",
)

app.include_router(mock_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": settings.VERSION}
