#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
================================================================
Copyright (C) 2019 * Ltd. All rights reserved.
PROJECT      :  Face_AI
FILE_NAME    :  sql_utils
AUTHOR       :  DAHAI LU
TIME         :  2021/7/16 下午4:13
PRODUCT_NAME :  PyCharm
================================================================
"""

import os.path as osp


def create_table(db_handler, sql_file):
    with open(sql_file, 'r') as fr:
        create_tb = str(fr.read())
        db_handler.cursor().execute(create_tb)


def remove_table(db_handler, table_name):
    try:
        cmd_sql = "drop table if exists {};".format(table_name)
        db_handler.cursor().execute(cmd_sql)
        db_handler.commit()
    except Exception as e:
        print(e.args)
        db_handler.rollback()


def update_table(db_handler, sql_file, reset=False):
    table_name = osp.splitext(osp.basename(sql_file))[0]
    if reset:
        remove_table(db_handler=db_handler, table_name=table_name)
    create_table(db_handler=db_handler, sql_file=sql_file)
