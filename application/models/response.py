# -*- coding: utf-8 -*-
# -----------------------------------
# @CreateTime   : 2020/8/7 2:19
# @Author       : Mark Shawn
# @Email        : shawninjuly@gmail.com
# ------------------------------------

"""
response model
"""

from typing import Union
from pydantic import BaseModel


class CRUD(BaseModel):
    data: Union[list, dict] = None
    message: Union[str, tuple] = None  # Exception的返回参数args是一个tuple

