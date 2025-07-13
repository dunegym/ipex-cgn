from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QTextEdit, QLineEdit, QWidget, QStackedWidget
from PyQt5.QtCore import pyqtSignal, QTimer, Qt
import threading
import time
import logging
import os
from config import LLM_MODEL_LIST, LLM_QUANTIZATION_LIST, LLM_DEVICE_LIST

class MainWindow(QMainWindow):
    console_signal = pyqtSignal(str)
    chat_signal = pyqtSignal(str)
    update_ui_signal = pyqtSignal(bool)
    
    def __init__(self, manager):
        super().__init__()
        self.manager = manager  # 传入聊天管理器
        self.loading_active = False  # 添加加载状态标志
        self.available_models = self.get_available_models()  # 获取可用的模型列表
        self.init_ui()  # 初始化用户界面
        # 连接信号到槽
        self.console_signal.connect(self.update_console)
        self.chat_signal.connect(self.update_chat)
        self.update_ui_signal.connect(self.update_ui_state)

    def get_available_models(self):
        """检查model文件夹下实际存在的模型"""
        available_models = []
        model_dir = os.path.join(os.path.dirname(__file__), 'model')
        
        # 如果model文件夹不存在，返回空列表
        if not os.path.exists(model_dir):
            logging.warning(f"模型文件夹不存在: {model_dir}")
            return available_models
        
        # 遍历配置文件中的模型列表
        for model_name in LLM_MODEL_LIST:
            model_path = os.path.join(model_dir, model_name)
            # 检查模型文件夹是否存在
            if os.path.exists(model_path) and os.path.isdir(model_path):
                available_models.append(model_name)
                logging.info(f"发现可用模型: {model_name}")
            else:
                logging.info(f"模型不存在: {model_name}")
        
        if not available_models:
            logging.warning("未找到任何可用模型")
        
        return available_models

    def init_ui(self):
        # 设置窗口标题和大小
        self.setWindowTitle("LLM 聊天助手")
        self.setGeometry(100, 100, 1600, 1200)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)

        font = self.font()
        font.setPointSize(int(font.pointSize() * 1.3))  # 字体放大0.3倍，确保为整数
        self.setFont(font)

        # 顶部导航栏
        nav_layout = QHBoxLayout()
        self.text_to_text_button = QPushButton("文生文")
        self.text_to_image_button = QPushButton("文生图")
        self.image_to_text_button = QPushButton("图生文")
        self.speech_rec_button = QPushButton("语音识别")
        
        nav_layout.addWidget(self.text_to_text_button)
        nav_layout.addWidget(self.text_to_image_button)
        nav_layout.addWidget(self.image_to_text_button)
        nav_layout.addWidget(self.speech_rec_button)
        
        main_layout.addLayout(nav_layout)

        # 用于切换页面的堆叠小部件
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # 创建页面
        self.text_to_text_page = self.create_text_to_text_page()
        self.text_to_image_page = self.create_placeholder_page("文生图功能开发中...")
        self.image_to_text_page = self.create_placeholder_page("图生文功能开发中...")
        self.speech_rec_page = self.create_placeholder_page("语音识别功能开发中...")

        # 将页面添加到堆叠小部件
        self.stacked_widget.addWidget(self.text_to_text_page)
        self.stacked_widget.addWidget(self.text_to_image_page)
        self.stacked_widget.addWidget(self.image_to_text_page)
        self.stacked_widget.addWidget(self.speech_rec_page)

        # 连接导航按钮
        self.text_to_text_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.text_to_image_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        self.image_to_text_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        self.speech_rec_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))

        # 设置默认页面
        self.stacked_widget.setCurrentIndex(0)

    def create_text_to_text_page(self):
        """创建文生文页面"""
        page_widget = QWidget()
        layout = QVBoxLayout(page_widget)

        # 模型选择部分
        model_layout = QHBoxLayout()
        layout.addLayout(model_layout)

        model_label = QLabel("选择模型:")
        model_layout.addWidget(model_label)

        self.model_combo = QComboBox()
        # 使用动态获取的可用模型列表
        if self.available_models:
            self.model_combo.addItems(self.available_models)
        else:
            self.model_combo.addItem("无可用模型")
            self.model_combo.setEnabled(False)
        model_layout.addWidget(self.model_combo)

        quant_label = QLabel("量化精度:")
        model_layout.addWidget(quant_label)

        self.quant_combo = QComboBox()
        self.quant_combo.addItems(LLM_QUANTIZATION_LIST)  # 添加量化精度列表
        model_layout.addWidget(self.quant_combo)

        device_label = QLabel("选择设备:")
        model_layout.addWidget(device_label)

        self.device_combo = QComboBox()
        self.device_combo.addItems(LLM_DEVICE_LIST)  # 添加设备列表
        model_layout.addWidget(self.device_combo)

        # 聊天显示部分
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)  # 设置为只读
        layout.addWidget(self.chat_display, 7)  # 添加伸缩因子为7
        self.chat_display.setStyleSheet("color: black;")

        # 按钮部分 - 移动到聊天框和控制台之间
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        self.load_unload_button = QPushButton("加载模型")
        self.load_unload_button.clicked.connect(self.toggle_model)  # 绑定新的切换事件
        button_layout.addWidget(self.load_unload_button)

        self.clear_button = QPushButton("清空上下文")
        self.clear_button.clicked.connect(self.do_clear_history)  # 绑定清空上下文事件
        self.clear_button.setEnabled(False)
        button_layout.addWidget(self.clear_button)
        
        # 刷新模型按钮 - 移动到这里
        self.refresh_button = QPushButton("刷新模型")
        self.refresh_button.clicked.connect(self.refresh_model_list)
        button_layout.addWidget(self.refresh_button)
        
        self.download_button = QPushButton("下载模型")
        button_layout.addWidget(self.download_button)

        # 控制台显示部分
        self.console_display = QTextEdit()
        self.console_display.setReadOnly(True)  # 设置为只读
        self.console_display.setStyleSheet("background-color: lightgray;")
        layout.addWidget(self.console_display, 3)  # 添加伸缩因子为3

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
        
        return page_widget

    def create_placeholder_page(self, text):
        """为正在开发的功能创建占位页面"""
        page_widget = QWidget()
        layout = QVBoxLayout(page_widget)
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        return page_widget
        
    def toggle_model(self):
        # 根据当前状态决定是加载还是卸载模型
        if self.manager.pipe is None:
            # 如果模型未加载，则加载模型
            self.do_load_model()
        else:
            # 如果模型已加载，则卸载模型
            self.do_unload_model()

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
        # 更新合并后按钮的文本和状态
        self.load_unload_button.setText("卸载模型" if model_loaded else "加载模型")
        self.load_unload_button.setEnabled(True)  # 操作完成后启用按钮
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
        # 首先禁用按钮，防止重复点击
        self.load_unload_button.setEnabled(False)
        self.load_unload_button.setText("正在加载...")
        self.console_signal.emit("开始加载模型......\n")
        
        # 设置一个标志来控制加载状态指示器
        self.loading_active = True
        
        # 实际加载模型的线程
        def load():
            success = False
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
        # 禁用按钮并更新文本
        self.load_unload_button.setEnabled(False)
        self.load_unload_button.setText("正在卸载...")
        
        # 卸载模型的逻辑
        self.manager.unload_model(self.console_callback)
        
        # 更新UI状态
        self.update_ui_signal.emit(False)  # 传递False表示模型已卸载
        
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

    def refresh_model_list(self):
        """刷新模型列表"""
        # 重新获取可用模型
        self.available_models = self.get_available_models()
        
        # 清空当前下拉框
        self.model_combo.clear()
        
        # 重新填充下拉框
        if self.available_models:
            self.model_combo.addItems(self.available_models)
            self.model_combo.setEnabled(True)
            self.console_callback(f"模型列表已刷新，找到 {len(self.available_models)} 个可用模型\n")
        else:
            self.model_combo.addItem("无可用模型")
            self.model_combo.setEnabled(False)
            self.console_callback("未找到任何可用模型，请检查model文件夹\n")

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
            QTimer.singleShot(100, lambda: QApplication.instance().quit())
        except Exception as e:
            # 记录异常但仍然退出
            logging.error(f"关闭程序时出错: {str(e)}")
            event.accept()