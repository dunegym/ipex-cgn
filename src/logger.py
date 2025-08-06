import os
import logging
import datetime

# --- Absolute Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

def setup_logger():
    """
    初始化日志记录器，创建日志文件并配置日志格式
    :return: 日志文件路径
    """
    # 创建logs目录（如果不存在）
    log_dir = os.path.join(project_root, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名，包含日期和时间
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"app_{current_time}.log")
    
    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除可能存在的旧处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件处理器，用于将日志写入文件
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器，用于在控制台输出日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器，定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file
