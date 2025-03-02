from pydantic.v1 import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv
import pathlib

# Get the directory containing config.py
BASE_DIR = pathlib.Path(__file__).parent

# Load .env file from the backend directory
load_dotenv(BASE_DIR / ".env")

class ServerConfig(BaseSettings):
    SERVER_PASSWORD: str

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True

def load_server_config() -> ServerConfig:
    """Load server configuration from environment variables"""
    config = ServerConfig()
    
    if not config.SERVER_PASSWORD:
        raise ValueError("SERVER_PASSWORD must be set in environment variables or .env file")
    
    print(f"Loaded server password from environment") # Debug line
        
    return config 