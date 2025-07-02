from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QTextEdit, QLineEdit, QWidget
from PyQt5.QtCore import Qt
import threading
import time
import os
from transformers import AutoTokenizer
import openvino_genai as ov_genai
from config import MODEL_LIST, QUANTIZATION_LIST, DEVICE_LIST
import logging
from PyQt5.QtCore import pyqtSignal
import datetime

# 初始化日志记录器
def setup_logger():
    # 创建logs目录（如果不存在）
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名，包含日期和时间
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"app_{current_time}.log")
    
    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file

# 设置日志记录
log_file_path = setup_logger()
logging.info("程序启动")
logging.info(f"日志保存路径: {log_file_path}")

# 定义一个管理LLM聊天的类
class LLMChatManager:
    def __init__(self):
        # 初始化管道和分词器
        self.pipe = None
        self.tokenizer = None
        self.chat_history = []  # 聊天历史记录

    def clear_history(self):
        # 清空聊天历史记录
        self.chat_history = []

    def load_model(self, model_name, quant, device, console_callback=None):
        # 加载指定的模型
        model_dir = f"model/{model_name}-{quant}"
        logging.info(f"开始加载模型: {model_name}, 量化精度: {quant}, 设备: {device}")
        if console_callback:
            console_callback('Loading......\n')
            # 添加短暂延时确保信号处理
            time.sleep(0.1)
            
        try:
            start_time = time.time()
            if console_callback:
                console_callback(f"正在准备模型目录: {model_dir}...\n")
                time.sleep(0.1)
                
            if device == 'NPU':
                # 如果设备是NPU，创建缓存目录
                if not os.path.exists('.npucache'):
                    os.makedirs('.npucache')
                if console_callback:
                    console_callback(f"为NPU设备创建缓存目录...\n")
                    time.sleep(0.1)
                
                # 加载前通知
                if console_callback:
                    console_callback(f"正在加载NPU模型，这可能需要一些时间...\n")
                    time.sleep(0.1)
                    
                self.pipe = ov_genai.LLMPipeline(
                    model_dir,
                    device,
                    CACHE_DIR=f".npucache/{model_name}-{quant}",
                    MAX_PROMPT_LEN=4096
                )
            else:
                # 如果设备不是NPU，直接加载模型
                if console_callback:
                    console_callback(f"正在加载模型到{device}设备，这可能需要一些时间...\n")
                    time.sleep(0.1)
                    
                self.pipe = ov_genai.LLMPipeline(model_dir, device)

            # 加载分词器
            if console_callback:
                console_callback("模型加载完成，正在加载分词器...\n")
                time.sleep(0.1)
                
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=True
            )
            end_time = time.time()
            load_time = end_time - start_time
            if console_callback:
                console_callback(f"模型加载成功！耗时：{load_time:.2f} 秒\n\n")
            logging.info(f"模型加载成功: {model_name}, 耗时: {load_time:.2f} 秒")
            return True
        except Exception as e:
            # 如果加载失败，清空管道和分词器
            self.pipe = None
            self.tokenizer = None
            self.handle_exception(e, console_callback, prefix="模型加载失败")
            return False

    def unload_model(self, console_callback=None):
        # 卸载模型
        logging.info("开始卸载模型")
        try:
            self.pipe = None
            self.tokenizer = None
            if console_callback:
                console_callback("模型已成功卸载！\n\n")
            logging.info("模型卸载成功")
            return True
        except Exception as e:
            self.handle_exception(e, console_callback, prefix="模型卸载失败")
            return False

    def build_prompt(self, user_input, model_name):
        # 构建聊天提示
        model_dir = None
        for quant in QUANTIZATION_LIST:
            candidate = f"model/{model_name}-{quant}"
            if os.path.isdir(candidate):
                model_dir = candidate
                break
        if model_dir is None:
            model_dir = f"model/{model_name}-int4"  # 默认使用int4量化模型

        try:
            n = len(self.chat_history)
            history = self.chat_history[-n:] if n <= 6 else self.chat_history[-6:]  # 限制历史记录长度为6
            messages = [
                {"role": "system", "content": "You are a helpful assistant. "}
            ] + history + [{"role": "user", "content": user_input}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            self.handle_exception(e, prefix="Prompt 构建失败")
            return f"<|user|>\n{user_input}\n<|assistant|>\n"

    def generate_reply(self, prompt, window, max_new_tokens=2048):
        # 生成回复
        if not self.pipe:
            err = RuntimeError("模型未加载")
            self.handle_exception(err)
            raise err
        try:
            device = getattr(self.pipe, 'device', None)
            do_sample = False if device and str(device).upper() == 'NPU' else True  # 根据设备类型设置采样方式
            streamer = lambda x: window.chat_callback(x)  # 定义流式输出回调
            result = self.pipe.generate(
                [prompt],
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True
            )
            perf_metrics = result.perf_metrics
            throughput = perf_metrics.get_throughput().mean if hasattr(perf_metrics, 'get_throughput') else '无法统计'
            logging.info(f"生成回复完成, 速度: %.2f tokens/s"%(throughput))
            return result
        except Exception as e:
            self.handle_exception(e, prefix="推理失败")
            raise

    def append_history(self, user_input, assistant_output):
        # 将用户输入和助手输出添加到聊天历史记录中
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": str(assistant_output)})
        if len(self.chat_history) > 6:
            self.chat_history = self.chat_history[-6:]  # 保持历史记录长度不超过6

    def handle_exception(self, e, console_callback=None, prefix="错误"):
        # 处理异常并记录日志
        msg = f"{prefix}: {str(e)}\n\n"
        logging.error(msg)
        if console_callback:
            console_callback(msg)

# 定义主窗口类
class MainWindow(QMainWindow):
    console_signal = pyqtSignal(str)
    chat_signal = pyqtSignal(str)
    update_ui_signal = pyqtSignal(bool)
    def __init__(self):
        super().__init__()
        self.manager = LLMChatManager()  # 初始化聊天管理器
        self.loading_active = False  # 添加加载状态标志
        self.init_ui()  # 初始化用户界面
        # 连接信号到槽
        self.console_signal.connect(self.update_console)
        self.chat_signal.connect(self.update_chat)
        self.update_ui_signal.connect(self.update_ui_state)

    def init_ui(self):
        # 设置窗口标题和大小
        self.setWindowTitle("LLM 聊天助手")
        self.setGeometry(100, 100, 1600, 1200)  # 宽度放大1倍，高度放大2倍

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        font = self.font()
        font.setPointSize(int(font.pointSize() * 1.3))  # 字体放大0.3倍，确保为整数
        self.setFont(font)

        # 模型选择部分
        model_layout = QHBoxLayout()
        layout.addLayout(model_layout)

        model_label = QLabel("选择模型:")
        model_layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_LIST)  # 添加模型列表
        model_layout.addWidget(self.model_combo)

        quant_label = QLabel("量化精度:")
        model_layout.addWidget(quant_label)

        self.quant_combo = QComboBox()
        self.quant_combo.addItems(QUANTIZATION_LIST)  # 添加量化精度列表
        model_layout.addWidget(self.quant_combo)

        device_label = QLabel("选择设备:")
        model_layout.addWidget(device_label)

        self.device_combo = QComboBox()
        self.device_combo.addItems(DEVICE_LIST)  # 添加设备列表
        model_layout.addWidget(self.device_combo)

        # 按钮部分
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        self.load_button = QPushButton("加载模型")
        self.load_button.clicked.connect(self.do_load_model)  # 绑定加载模型事件
        button_layout.addWidget(self.load_button)

        self.unload_button = QPushButton("卸载模型")
        self.unload_button.clicked.connect(self.do_unload_model)  # 绑定卸载模型事件
        self.unload_button.setEnabled(False)
        button_layout.addWidget(self.unload_button)

        self.clear_button = QPushButton("清空上下文")
        self.clear_button.clicked.connect(self.do_clear_history)  # 绑定清空上下文事件
        self.clear_button.setEnabled(False)
        button_layout.addWidget(self.clear_button)

        # 聊天显示部分
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)  # 设置为只读
        layout.addWidget(self.chat_display)
        self.chat_display.setStyleSheet("color: black;")

        # 控制台显示部分
        self.console_display = QTextEdit()
        self.console_display.setReadOnly(True)  # 设置为只读
        self.console_display.setStyleSheet("background-color: lightgray;")
        layout.addWidget(self.console_display)

        # 用户输入部分
        input_layout = QHBoxLayout()
        layout.addLayout(input_layout)

        self.user_input = QLineEdit()
        self.user_input.textChanged.connect(self.update_send_button)  # 绑定输入框事件
        input_layout.addWidget(self.user_input)

        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.do_send_message)  # 绑定发送消息事件
        self.send_button.setEnabled(False)
        input_layout.addWidget(self.send_button)
        
        # 添加日志文件路径显示
        log_path_layout = QHBoxLayout()
        layout.addLayout(log_path_layout)
        
        log_path_label = QLabel("日志文件:")
        log_path_layout.addWidget(log_path_label)
        
        self.log_path_display = QLineEdit()
        self.log_path_display.setText(log_file_path)
        self.log_path_display.setReadOnly(True)
        log_path_layout.addWidget(self.log_path_display)
        
        open_log_button = QPushButton("打开日志目录")
        open_log_button.clicked.connect(self.open_log_directory)
        log_path_layout.addWidget(open_log_button)

    # 添加一个打开日志目录的方法
    def open_log_directory(self):
        log_dir = os.path.dirname(log_file_path)
        if os.path.exists(log_dir):
            # 在Windows上打开文件夹
            os.startfile(log_dir)
        else:
            self.console_callback("日志目录不存在\n")
        
    def update_console(self, msg):
        self.console_display.append(msg)
        # 确保滚动到最新内容
        self.console_display.verticalScrollBar().setValue(
            self.console_display.verticalScrollBar().maximum()
        )
        # 强制处理事件，立即更新UI
        QApplication.processEvents()

    def update_chat(self, msg):
        self.chat_display.insertPlainText(msg)

    def update_ui_state(self, model_loaded):
        self.load_button.setEnabled(not model_loaded)
        self.unload_button.setEnabled(model_loaded)
        self.clear_button.setEnabled(model_loaded)
        self.user_input.setEnabled(model_loaded)
        self.update_send_button()

    def console_callback(self, msg):
        self.console_signal.emit(msg)

    def chat_callback(self, msg):
        self.chat_signal.emit(msg)

    def update_send_button(self):
        # 更新发送按钮的状态
        is_enabled = bool(self.user_input.text().strip()) and self.manager.pipe is not None
        self.send_button.setEnabled(is_enabled)
        self.user_input.setEnabled(self.manager.pipe is not None)
        self.clear_button.setEnabled(self.manager.pipe is not None)

    def do_load_model(self):
        # 首先禁用相关按钮，防止重复点击
        self.load_button.setEnabled(False)
        self.console_signal.emit("开始加载模型......\n")
        
        # 设置一个标志来控制加载状态指示器
        self.loading_active = True
        
        # 创建一个独立线程来显示加载状态
        def loading_indicator():
            pass  # 删除模型加载中提示代码

        # 启动加载状态指示器线程
        indicator_thread = threading.Thread(target=loading_indicator, daemon=True)
        indicator_thread.start()
        
        # 实际加载模型的线程
        def load():
            try:
                selected_model = self.model_combo.currentText()
                selected_quant = self.quant_combo.currentText()
                selected_device = self.device_combo.currentText()
                
                # 使用修改后的回调函数
                success = self.manager.load_model(
                    selected_model, 
                    selected_quant, 
                    selected_device, 
                    lambda msg: self.console_signal.emit(msg)
                )
            finally:
                # 无论成功与否，都停止加载状态指示器
                self.loading_active = False
                # 在主线程中更新UI状态
                self.update_ui_signal.emit(success)
        
        threading.Thread(target=load, daemon=True).start()
    def do_unload_model(self):
        # 卸载模型的逻辑
        self.manager.unload_model(self.console_callback)
        self.load_button.setEnabled(True)
        self.unload_button.setEnabled(False)
        self.update_send_button()

    def do_clear_history(self):
        # 清空聊天历史记录的逻辑
        self.manager.clear_history()
        self.console_callback("上下文已清空\n\n")
        self.update_send_button()

    def do_send_message(self):
        # 发送消息的逻辑
        def send():
            user_input = self.user_input.text()
            self.chat_callback(f"\n\n用户: \n{user_input}\n")
            self.user_input.clear()
            self.send_button.setEnabled(False)
            self.console_callback("消息成功发送，等待输出中......\n")
            try:
                selected_model = self.model_combo.currentText()
                prompt = self.manager.build_prompt(user_input, selected_model)
                self.chat_callback("\n助手: \n")
                result = self.manager.generate_reply(prompt, self)
                perf_metrics = result.perf_metrics
                self.console_callback(f"已成功输出，速度为 {perf_metrics.get_throughput().mean:.2f} tokens/s\n\n")
                self.manager.append_history(user_input, result)
            except Exception as e:
                self.chat_callback(f"助手: 无法生成回复，错误: {str(e)}\n\n")
            finally:
                self.update_send_button()

        threading.Thread(target=send, daemon=True).start()

    def closeEvent(self, event):
        # 在程序关闭前卸载模型并释放资源
        try:
            # 设置标志，终止所有正在运行的线程
            self.loading_active = False
            
            # 如果模型已加载，则卸载模型
            if self.manager.pipe is not None:
                self.console_callback("程序正在关闭，卸载模型并释放资源......\n")
                # 直接在主线程中卸载模型，避免使用新线程
                self.manager.unload_model(self.console_callback)
                
            self.console_callback("资源已释放，程序已安全退出。\n")
            logging.info("程序关闭，资源已释放。")
            
            # 确保所有待处理的事件都被处理
            QApplication.processEvents()
            
            # 接受关闭事件
            event.accept()
            
            # 使用定时器延迟退出应用程序
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, lambda: QApplication.instance().quit())
        except Exception as e:
            # 记录异常但仍然退出
            logging.error(f"关闭程序时出错: {str(e)}")
            event.accept()
            
# 程序入口
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    
    # 设置应用程序退出选项，确保应用完全退出
    app.setQuitOnLastWindowClosed(True)
    import sys
    try:
        # 使用退出代码
        sys.exit(app.exec_())
    finally:
        # 确保程序完全退出
        import os
        os._exit(0)