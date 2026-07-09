from __future__ import annotations

import logging
import os
import sys

import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import create_app
from api.config import ApiConfig

app = create_app()

if __name__ == "__main__":
    config = ApiConfig.from_env()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format=config.log_format,
        datefmt=config.log_datefmt,
    )
    uvicorn.run(
        "api.main:app",
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        log_level=config.log_level.lower(),
        reload=config.api_workers <= 1,
    )
