from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables or .env file.
    """
    APP_NAME: str = "Constructor Agent Tools"
    VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # Constructor API settings
    CONSTRUCTOR_API_URL: str = "http://localhost:8001/mock-constructor"
    CONSTRUCTOR_API_KEY: Optional[str] = None
    
    # Gemini LLM settings
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    
    # Bundle Agent Settings
    BUNDLE_SOLVER_TIMEOUT_SECONDS: float = 10.0
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

settings = Settings()

