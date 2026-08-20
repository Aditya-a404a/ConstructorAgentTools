from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from constructor_agent_tools.settings import settings
from constructor_agent_tools.mock_constructor.server import router as mock_router
from constructor_agent_tools.bundle.server import router as bundle_router
from constructor_agent_tools.searchandising.server import router as searchandising_router

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Advanced autonomous agent systems built on Constructor.io APIs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mock_router)
app.include_router(bundle_router)
app.include_router(searchandising_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": settings.VERSION}

