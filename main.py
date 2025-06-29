from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QTextEdit, QLineEdit, QWidget
from PyQt5.QtCore import Qt
import threading
import logging
import time
import os
from transformers import AutoTokenizer
import openvino_genai as ov_genai

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

MODEL_LIST = ['TinyLlama-1.1B', 'DeepSeek-1.5B']
QUANTIZATION_LIST = ['int4', 'int8']
DEVICE_LIST = ['CPU', 'GPU', 'NPU']

class LLMChatManager:
    def __init__(self):
        self.pipe = None
        self.tokenizer = None
        self.chat_history = []

    def clear_history(self):
        self.chat_history = []

    def load_model(self, model_name, quant, device, console_callback=None):
        model_dir = f"model/{model_name}-{quant}"
        if console_callback:
            console_callback('Loading......\n')
        try:
            start_time = time.time()
            if device == 'NPU':
                if not os.path.exists('.npucache'):
                    os.makedirs('.npucache')
                self.pipe = ov_genai.LLMPipeline(
                    model_dir,
                    device,
                    CACHE_DIR=f".npucache/{model_name}-{quant}",
                    MAX_PROMPT_LEN=2048
                )
            else:
                self.pipe = ov_genai.LLMPipeline(model_dir, device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=True
            )
            end_time = time.time()
            load_time = end_time - start_time
            if console_callback:
                console_callback(f"模型加载成功！耗时：{load_time:.2f} 秒\n\n")
            logging.info(f"模型 {model_name}-{quant} 加载成功，设备: {device}，耗时: {load_time:.2f} 秒")
            return True
        except Exception as e:
            self.pipe = None
            self.tokenizer = None
            self.handle_exception(e, console_callback, prefix="模型加载失败")
            return False

    def unload_model(self, console_callback=None):
        try:
            self.pipe = None
            self.tokenizer = None
            if console_callback:
                console_callback("模型已成功卸载！\n\n")
            logging.info("模型已成功卸载！")
            return True
        except Exception as e:
            self.handle_exception(e, console_callback, prefix="模型卸载失败")
            return False

    def build_prompt(self, user_input, model_name):
        model_dir = None
        for quant in QUANTIZATION_LIST:
            candidate = f"model/{model_name}-{quant}"
            if os.path.isdir(candidate):
                model_dir = candidate
                break
        if model_dir is None:
            model_dir = f"model/{model_name}-int4"  # 兜底

        try:
            n = len(self.chat_history)
            history = self.chat_history[-n:] if n <= 6 else self.chat_history[-6:]
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

    def generate_reply(self, prompt, max_new_tokens=512):
        if not self.pipe:
            err = RuntimeError("模型未加载")
            self.handle_exception(err)
            raise err
        try:
            device = getattr(self.pipe, 'device', None)
            do_sample = False if device and str(device).upper() == 'NPU' else True
            result = self.pipe.generate(
                [prompt],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True
            )
            logging.info(f"推理完成，tokens: {max_new_tokens}, do_sample: {do_sample}")
            return result
        except Exception as e:
            self.handle_exception(e, prefix="推理失败")
            raise

    def append_history(self, user_input, assistant_output):
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": str(assistant_output)})
        if len(self.chat_history) > 6:
            self.chat_history = self.chat_history[-6:]

    def handle_exception(self, e, console_callback=None, prefix="错误"):
        msg = f"{prefix}: {str(e)}\n\n"
        logging.error(msg)
        if console_callback:
            console_callback(msg)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.manager = LLMChatManager()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("LLM 聊天助手")
        self.setGeometry(100, 100, 1600, 1200)  # 宽度放大1倍，高度放大2倍

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        font = self.font()
        font.setPointSize(int(font.pointSize() * 1.3))  # 字体放大0.3倍，确保为整数
        self.setFont(font)

        # Model selection
        model_layout = QHBoxLayout()
        layout.addLayout(model_layout)

        model_label = QLabel("选择模型:")
        model_layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_LIST)
        model_layout.addWidget(self.model_combo)

        quant_label = QLabel("量化精度:")
        model_layout.addWidget(quant_label)

        self.quant_combo = QComboBox()
        self.quant_combo.addItems(QUANTIZATION_LIST)
        model_layout.addWidget(self.quant_combo)

        device_label = QLabel("选择设备:")
        model_layout.addWidget(device_label)

        self.device_combo = QComboBox()
        self.device_combo.addItems(DEVICE_LIST)
        model_layout.addWidget(self.device_combo)

        # Buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        self.load_button = QPushButton("加载模型")
        self.load_button.clicked.connect(self.do_load_model)
        button_layout.addWidget(self.load_button)

        self.unload_button = QPushButton("卸载模型")
        self.unload_button.clicked.connect(self.do_unload_model)
        self.unload_button.setEnabled(False)
        button_layout.addWidget(self.unload_button)

        self.clear_button = QPushButton("清空上下文")
        self.clear_button.clicked.connect(self.do_clear_history)
        self.clear_button.setEnabled(False)
        button_layout.addWidget(self.clear_button)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        # Console display
        self.console_display = QTextEdit()
        self.console_display.setReadOnly(True)
        self.console_display.setStyleSheet("background-color: lightgray;")
        layout.addWidget(self.console_display)

        # User input
        input_layout = QHBoxLayout()
        layout.addLayout(input_layout)

        self.user_input = QLineEdit()
        self.user_input.textChanged.connect(self.update_send_button)
        input_layout.addWidget(self.user_input)

        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.do_send_message)
        self.send_button.setEnabled(False)
        input_layout.addWidget(self.send_button)

    def console_callback(self, msg):
        self.console_display.append(msg)

    def chat_callback(self, msg):
        self.chat_display.append(msg)

    def update_send_button(self):
        is_enabled = bool(self.user_input.text().strip()) and self.manager.pipe is not None
        self.send_button.setEnabled(is_enabled)
        self.user_input.setEnabled(self.manager.pipe is not None)
        self.clear_button.setEnabled(self.manager.pipe is not None)

    def do_load_model(self):
        def load():
            selected_model = self.model_combo.currentText()
            selected_quant = self.quant_combo.currentText()
            selected_device = self.device_combo.currentText()
            self.load_button.setEnabled(False)
            self.console_callback("开始加载模型......\n")
            success = self.manager.load_model(selected_model, selected_quant, selected_device, self.console_callback)
            self.load_button.setEnabled(not success)
            self.unload_button.setEnabled(success)
            self.update_send_button()

        threading.Thread(target=load, daemon=True).start()

    def do_unload_model(self):
        self.manager.unload_model(self.console_callback)
        self.load_button.setEnabled(True)
        self.unload_button.setEnabled(False)
        self.update_send_button()

    def do_clear_history(self):
        self.manager.clear_history()
        self.console_callback("上下文已清空\n\n")
        self.update_send_button()

    def do_send_message(self):
        def send():
            user_input = self.user_input.text()
            if user_input.strip().lower() == 'quit':
                self.close()
                return
            self.chat_callback(f"用户: {user_input}\n\n")
            self.user_input.clear()
            self.send_button.setEnabled(False)
            self.console_callback("消息成功发送，等待输出中......\n")
            try:
                selected_model = self.model_combo.currentText()
                prompt = self.manager.build_prompt(user_input, selected_model)
                result = self.manager.generate_reply(prompt)
                perf_metrics = result.perf_metrics
                self.chat_callback(f"助手: {result}\n\n")
                self.console_callback(f"已成功输出，速度为 {perf_metrics.get_throughput().mean:.2f} tokens/s\n\n")
                self.manager.append_history(user_input, result)
            except Exception as e:
                self.chat_callback(f"助手: 无法生成回复，错误: {str(e)}\n\n")
            finally:
                self.update_send_button()

        threading.Thread(target=send, daemon=True).start()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
