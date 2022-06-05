#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 17:45
# @Author  : CoderCharm
# @File    : __init__.py
# @Software: PyCharm
# @Github  : github/CoderCharm
# @Email   : wg_python@163.com
# @Desc    :
"""

版本路由区分

# 可以在这里添加所需要的依赖
https://fastapi.tiangolo.com/tutorial/bigger-applications/#import-fastapi

fastapi 没有像flask那样 分组子路由没有 middleware("http") 但是有 dependencies

"""

from fastapi import APIRouter, Depends

from apphelper.utils import settings
from apphelper.common.deps import check_authority
from application.api.user_router import router as auth_router
from application.api.items_router import router as items_router
from application.api.api_router import router as sys_api_router
from application.api.casbin_router import router as sys_casbin_router
from application.api.scheduler_router import router as scheduler_router
from application.api.doc_router import router as doc_router
from application.api.face_router import router as face_router


api_v1_router = APIRouter()
api_v1_router.include_router(doc_router)
api_v1_router.include_router(auth_router, prefix="/admin/auth", tags=["用户"])

# api_v1_router.include_router(items_router, tags=["测试API"], dependencies=[Depends(check_jwt_token)])
# check_authority 权限验证内部包含了 token 验证 如果不校验权限可直接 dependencies=[Depends(check_jwt_token)]
if settings.DEBUG:
    api_v1_router.include_router(items_router, tags=["测试API"], dependencies=[Depends(check_authority)])
api_v1_router.include_router(scheduler_router, tags=["任务调度"],  dependencies=[Depends(check_authority)])
api_v1_router.include_router(sys_api_router, tags=["服务API管理"],  dependencies=[Depends(check_authority)])
api_v1_router.include_router(sys_casbin_router, tags=["权限API管理"],  dependencies=[Depends(check_authority)])
api_v1_router.include_router(face_router, tags=["人脸API管理"],  dependencies=[Depends(check_authority)])
