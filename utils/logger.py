# utils/logger.py
import os
import sys
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "Training",
    log_dir: str = "logs",
    log_level: str = "INFO",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_to_file: bool = True,
    log_to_console: bool = True
):
    """
    设置日志系统，同时输出到控制台和文件
    
    Args:
        name: logger 名称
        log_dir: 日志保存目录
        log_level: 全局日志级别
        console_level: 控制台输出级别
        file_level: 文件输出级别
        log_to_file: 是否输出到文件
        log_to_console: 是否输出到控制台
    
    Returns:
        logger: 配置好的 logger 对象
    
    Usage:
        from utils.logger import setup_logger
        
        logger = setup_logger("MyTrainer", log_dir="logs")
        logger.info("训练开始")
        logger.debug("详细调试信息")  # 只在文件里，控制台不显示
    """
    # 创建日志目录
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
    
    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()  # 清除已有的 handlers，防止重复
    
    # 日志格式
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 文件 handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name.lower()}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 同时创建一个 symlink 指向最新日志
        latest_link = os.path.join(log_dir, f"{name.lower()}_latest.log")
        try:
            if os.path.exists(latest_link):
                os.remove(latest_link)
            os.symlink(log_file, latest_link)
        except OSError:
            pass  # Windows 可能需要管理员权限
    
    # 控制台 handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str = "Training"):
    """
    获取已存在的 logger，如果不存在则创建一个简单的
    
    Args:
        name: logger 名称
    
    Returns:
        logger 对象
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # 如果没有 handler，创建一个简单的控制台 logger
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
        )
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    
    return logger