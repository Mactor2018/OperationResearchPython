# logger.py

import logging
import sys

# 创建一个模块级的 logger 实例
logger = logging.getLogger("OR_Python")
logger.setLevel(logging.DEBUG)  # 设置最低记录级别为 DEBUG，可根据需要修改为 INFO、WARNING 等

# 创建日志格式
formatter = logging.Formatter(
    fmt="(%(name)s:%(levelname)s)[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 创建控制台输出的 handler（StreamHandler 输出到终端）
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)  # 控制台输出的日志级别
console_handler.setFormatter(formatter)

# 避免重复添加 handler
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# 可选：防止日志被向上传播到 root logger，避免重复打印
logger.propagate = False
