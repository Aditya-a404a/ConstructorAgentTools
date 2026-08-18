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
    CONSTRUCTOR_API_URL: str = "http://localhost:8002/mock-constructor"
    CONSTRUCTOR_API_KEY: Optional[str] = None
    
    # Bundle Agent Settings
    BUNDLE_SOLVER_TIMEOUT_SECONDS: float = 10.0
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
