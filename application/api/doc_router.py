# @Time: 2022/5/31 9:13
# @Author: chang liu
# @Email: chang_liu_tamu@gmail.com
# @File:doc_router.py

#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
================================================================
Copyright (C) 2019 * Ltd. All rights reserved.
PROJECT      :  Face_AI
FILE_NAME    :  doc_router
AUTHOR       :  DAHAI LU
TIME         :  2021/7/16 下午4:56
PRODUCT_NAME :  PyCharm
================================================================
"""

import os
from fastapi import APIRouter
from starlette.responses import RedirectResponse
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from application.utils import settings

router = APIRouter()


@router.get('/', include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url=settings.DOCS_URL)


@router.get(settings.DOCS_URL, include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=settings.OPENAPI_URL,
        title=settings.TITLE + " - Swagger UI",
        oauth2_redirect_url=settings.OAUTH2_REDIRECT_URL,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_favicon_url='/static/favicon.png'
    )


@router.get(settings.OAUTH2_REDIRECT_URL, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@router.get(settings.REDOC_URL, include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=settings.OPENAPI_URL,
        title=settings.TITLE + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )


@router.get('/info', tags=['Utility'])
def info():
    """
    Enslist container configuration.
    """

    about = dict(
        version=settings.configs.version,
        tensorrt_version=os.getenv('TRT_VERSION', os.getenv('TENSORRT_VERSION')),
        log_level=settings.configs.log_level,
        models=vars(settings.configs.models),
        defaults=vars(settings.configs.defaults),
    )
    about['models'].pop('ga_ignore', None)
    about['models'].pop('rec_ignore', None)
    about['models'].pop('device', None)
    return about