#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 19:37
# @Author  : CoderCharm
# @File    : sys_authority.py
# @Software: PyCharm
# @Github  : github/CoderCharm
# @Email   : wg_python@163.com
# @Desc    :
"""

管理员角色的CRUD

"""
from loguru import logger
from sqlalchemy.orm import Session

from apphelper.service.curd_base import CRUDBase
from application.models.sys_auth import SysAuthorities
from apphelper.schemas.request import sys_authority_schema


class CRUDAuthorities(
    CRUDBase[SysAuthorities,
             sys_authority_schema.AuthorityCreate,
             sys_authority_schema.AuthorityUpdate]
):

    def create(self, db: Session, *, obj_in: sys_authority_schema.AuthorityCreate) -> SysAuthorities:
        """
        创建角色
        :param db:
        :param obj_in:
        :return:
        """
        db_obj = SysAuthorities(
            authority_id=obj_in.authority_id,
            authority_name=obj_in.authority_name,
            parent_id=obj_in.parent_id
        )
        try:
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
        except Exception as e:
            db.rollback()
            db_obj.parent_id = -1
            logger.warning(e.args)
        return db_obj


curd_authority = CRUDAuthorities(SysAuthorities)
