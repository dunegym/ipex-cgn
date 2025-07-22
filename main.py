from PyQt5.QtWidgets import QApplication  # 确保这行在开头
import logging
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
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 设置应用程序退出选项
    app.setQuitOnLastWindowClosed(True)
    
    try:
        # 启动事件循环
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        logging.info("用户中断程序")
    except Exception as e:
        logging.error(f"程序异常退出: {str(e)}")
    finally:
        logging.info("程序退出")