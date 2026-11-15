"""
auth.py — Simple API key authentication for state-changing endpoints (Fix #16).

Not full OAuth/JWT — disproportionate scope for a BCA project.
Reads API key from MLOPS_API_KEY environment variable or uses dev default.
"""

from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import API_KEY


# FastAPI security scheme — expects "X-API-Key" header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency for FastAPI routes that need authentication.
    Raises 401 if no key provided, 403 if key is wrong.

    Usage:
        @app.post("/api/approve", dependencies=[Depends(require_api_key)])
        async def approve(): ...
    """
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Set the X-API-Key header.",
        )
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )
    return api_key
