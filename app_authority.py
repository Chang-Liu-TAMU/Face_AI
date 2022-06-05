#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 16:24
# @Author  : CoderCharm
# @File    : main.py
# @Software: PyCharm
# @Github  : github/CoderCharm
# @Email   : wg_python@163.com
# @Desc    :

"""
pip install uvicorn
# 推荐启动方式 main指当前文件名字 app指FastAPI实例化后对象名称
uvicorn app_authority:app --host=127.0.0.1 --port=9997 --reload

类似flask 工厂模式创建
# 生产启动命令 去掉热重载 (可用supervisor托管后台运行)

在app_authority.py同文件下下启动
uvicorn app_authority:app --host=127.0.0.1 --port=9997 --workers=4

# 同样可以也可以配合gunicorn多进程启动  app_authority.py同文件下下启动 默认127.0.0.1:9997端口
gunicorn app_authority:app -w 4 -k uvicorn.workers.UvicornWorker -b 127.0.0.1:9997

"""

from apphelper.utils import settings
from application.server import create_app

# Read runtime settings from environment variables
app = create_app()

if __name__ == "__main__":
    import uvicorn

    # 输出所有的路由
    for route in app.routes:
        if settings.DEBUG and hasattr(route, "methods"):
            print({'path': route.path, 'name': route.name, 'methods': route.methods})

    uvicorn.run(app="app_authority:app",
                host=settings.configs.host,
                port=settings.configs.port,
                workers=settings.configs.num_workers,
                reload=settings.configs.reload,
                debug=settings.configs.debug)