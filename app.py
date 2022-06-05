# @Time: 2022/5/17 14:37
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:app.py


import os
import uvicorn
from loguru import logger
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)
from core.env_parser import EnvConfigs
from application.utils.custom_log import init_logger
from application.api.face_router import router as face_router
from application.api.doc_router import router as doc_router

# Read runtime settings from environment variables
configs = EnvConfigs()

init_logger(configs=configs, logger=logger)

__version__ = configs.version
# with open('README.md', 'r', encoding='utf-8') as f:
#     openapi_desc = f.read()

app = FastAPI(
    title="Face-AI",
    description="FastAPI wrapper for Face AI.",
    # description=openapi_desc,
    version=__version__,
    docs_url=None,
    redoc_url=None
)

app.include_router(face_router)
app.include_router(doc_router)


# app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == '__main__':
    uvicorn.run(app="app:app",
                host=configs.host,
                port=configs.port,
                workers=configs.num_workers,
                reload=configs.reload,
                debug=configs.debug)
