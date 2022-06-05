#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/29 15:44
# @Author  : CoderCharm
# @File    : sys_api.py
# @Software: PyCharm
# @Github  : github/CoderCharm
# @Email   : wg_python@163.com
# @Desc    :

"""
token model
"""
from apphelper.utils import settings
from pydantic import BaseModel, Field


class TokenModel(BaseModel):
    token: str = Field(...)
    token_type: str = Field(settings.TOKEN_TYPE)
