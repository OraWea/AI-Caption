#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask应用启动脚本 - 生产模式（不重启）
"""
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("启动AI字幕翻译应用（生产模式）...")
    # 禁用调试模式，避免文件更改时重启
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
