from PyQt5.QtWidgets import QApplication
import logging
import os
import sys

# 导入自定义模块
from logger import setup_logger
from ui import MainWindow

# 主程序入口
if __name__ == "__main__":
    # 设置日志记录
    log_file_path = setup_logger()
    logging.info("程序启动")
    logging.info(f"日志保存路径: {log_file_path}")
    
    # 创建应用程序
    app = QApplication([])
    
    # 创建主窗口并传入管理器
    window = MainWindow()
    window.show()
    
    # 设置应用程序退出选项，确保应用完全退出
    app.setQuitOnLastWindowClosed(True)
    
    try:
        # 使用退出代码
        sys.exit(app.exec_())
    finally:
        # 确保程序完全退出
        os._exit(0)