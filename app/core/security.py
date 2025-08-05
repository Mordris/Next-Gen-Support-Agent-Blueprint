# app/core/security.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

# Define the header we expect to receive
api_key_header = APIKeyHeader(name="X-API-KEY")


async def api_key_auth(api_key: str = Security(api_key_header)):
    """
    Checks if the provided API key is valid.
    """
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
