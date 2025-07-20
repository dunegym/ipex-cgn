from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QTextEdit, QLineEdit, QWidget, QStackedWidget, QSpinBox, QScrollArea
from PyQt5.QtCore import pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QPixmap
import threading
import logging
import os
from config import LLM_MODEL_LIST, LLM_QUANTIZATION_LIST, LLM_DEVICE_LIST, T2I_MODEL_LIST, T2I_QUANTIZATION_LIST
from manager import LLMChatManager, T2IManager

class MainWindow(QMainWindow):
    llm_console_signal = pyqtSignal(str)     # 改名：LLM控制台信号
    llm_chat_signal = pyqtSignal(str)        # 改名：LLM聊天信号
    llm_update_ui_signal = pyqtSignal(bool)  # 改名：LLM UI更新信号
    # T2I相关信号保持不变
    t2i_console_signal = pyqtSignal(str)
    t2i_update_ui_signal = pyqtSignal(bool)
    t2i_image_signal = pyqtSignal(list)  # 用于显示生成的图像
    # 添加进度信号到类级别
    t2i_progress_signal = pyqtSignal(int, int)  # current_step, total_steps
    
    def __init__(self):
        super().__init__()
        self.llm_manager = LLMChatManager()  
        self.t2i_manager = T2IManager()      
        self.llm_loading_active = False      
        self.t2i_loading_active = False     # T2I加载状态标志
        self.llm_available_models = self.ui_get_available_llm_models()  
        self.t2i_available_models = self.ui_get_available_t2i_models()  
        
        # 添加进度相关变量
        self.t2i_progress_timer = None
        self.t2i_current_step = 0
        self.t2i_total_steps = 0
        
        self.ui_init()  
        # 连接信号到槽
        self.llm_console_signal.connect(self.ui_update_llm_console)  
        self.llm_chat_signal.connect(self.ui_update_llm_chat)        
        self.llm_update_ui_signal.connect(self.ui_update_llm_state)  
        # 连接文生图信号到槽
        self.t2i_console_signal.connect(self.ui_update_t2i_console)
        self.t2i_update_ui_signal.connect(self.ui_update_t2i_state)
        self.t2i_image_signal.connect(self.ui_update_t2i_images)
        
        # 连接进度信号
        self.t2i_progress_signal.connect(self.ui_update_t2i_progress)

    def ui_get_available_llm_models(self):  # 改名：方法名更明确
        """检查model文件夹下实际存在的LLM模型"""
        llm_available_models = []  # 改名：变量名更明确
        model_dir = os.path.join(os.path.dirname(__file__), 'model')
        
        # 如果model文件夹不存在，返回空列表
        if not os.path.exists(model_dir):
            logging.warning(f"[LLM] 模型文件夹不存在: {model_dir}")
            return llm_available_models
        
        # 遍历配置文件中的LLM模型列表
        for llm_model_name in LLM_MODEL_LIST:  # 改名：变量名更明确
            llm_model_path = os.path.join(model_dir, llm_model_name)  # 改名
            # 检查模型文件夹是否存在
            if os.path.exists(llm_model_path) and os.path.isdir(llm_model_path):
                llm_available_models.append(llm_model_name)
                logging.info(f"[LLM] 发现可用LLM模型: {llm_model_name}")
            else:
                logging.info(f"[LLM] LLM模型不存在: {llm_model_name}")
        
        if not llm_available_models:
            logging.warning("[LLM] 未找到任何可用LLM模型")
        else:
            logging.info(f"[LLM] 总共找到 {len(llm_available_models)} 个可用LLM模型")
        
        return llm_available_models

    def ui_get_available_t2i_models(self):
        """检查model文件夹下实际存在的文生图模型"""
        t2i_available_models = []  # 改名：变量名更明确
        model_dir = os.path.join(os.path.dirname(__file__), 'model')
        
        # 如果model文件夹不存在，返回空列表
        if not os.path.exists(model_dir):
            logging.warning(f"[T2I] 模型文件夹不存在: {model_dir}")
            return t2i_available_models
        
        # 遍历配置文件中的文生图模型列表
        for t2i_model_name in T2I_MODEL_LIST:  # 改名：变量名更明确
            t2i_model_path = os.path.join(model_dir, t2i_model_name)  # 改名
            # 检查模型文件夹是否存在
            if os.path.exists(t2i_model_path) and os.path.isdir(t2i_model_path):
                t2i_available_models.append(t2i_model_name)
                logging.info(f"[T2I] 发现可用文生图模型: {t2i_model_name}")
            else:
                logging.info(f"[T2I] 文生图模型不存在: {t2i_model_name}")
        
        if not t2i_available_models:
            logging.warning("[T2I] 未找到任何可用文生图模型")
        else:
            logging.info(f"[T2I] 总共找到 {len(t2i_available_models)} 个可用文生图模型")
        
        return t2i_available_models

    def ui_init(self):
        # 设置窗口标题和大小
        self.setWindowTitle("OpenVINO多功能AI平台")
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
        self.text_to_text_page = self.ui_create_llm_page()
        self.text_to_image_page = self.ui_create_t2i_page()  # 使用新的文生图页面
        self.image_to_text_page = self.ui_create_placeholder_page("图生文功能开发中...")
        self.speech_rec_page = self.ui_create_placeholder_page("语音识别功能开发中...")

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

    def ui_create_llm_page(self):
        """创建文生文页面"""
        llm_page_widget = QWidget()  # 改名
        llm_layout = QVBoxLayout(llm_page_widget)  # 改名

        # 模型选择部分
        llm_model_layout = QHBoxLayout()  # 改名
        llm_layout.addLayout(llm_model_layout)

        llm_model_label = QLabel("选择模型:")  # 改名
        llm_model_layout.addWidget(llm_model_label)

        self.llm_model_combo = QComboBox()  # 改名
        # 使用动态获取的可用模型列表
        if self.llm_available_models:  # 改名
            self.llm_model_combo.addItems(self.llm_available_models)
        else:
            self.llm_model_combo.addItem("无可用模型")
            self.llm_model_combo.setEnabled(False)
        llm_model_layout.addWidget(self.llm_model_combo)

        llm_quant_label = QLabel("量化精度:")  # 改名
        llm_model_layout.addWidget(llm_quant_label)

        self.llm_quant_combo = QComboBox()  # 改名
        self.llm_quant_combo.addItems(LLM_QUANTIZATION_LIST)
        llm_model_layout.addWidget(self.llm_quant_combo)

        llm_device_label = QLabel("选择设备:")  # 改名
        llm_model_layout.addWidget(llm_device_label)

        self.llm_device_combo = QComboBox()  # 改名
        self.llm_device_combo.addItems(LLM_DEVICE_LIST)
        llm_model_layout.addWidget(self.llm_device_combo)

        # 聊天显示部分
        self.llm_chat_display = QTextEdit()  # 改名
        self.llm_chat_display.setReadOnly(True)
        llm_layout.addWidget(self.llm_chat_display, 7)
        self.llm_chat_display.setStyleSheet("color: black;")

        # 按钮部分
        llm_button_layout = QHBoxLayout()  # 改名
        llm_layout.addLayout(llm_button_layout)

        self.llm_load_unload_button = QPushButton("加载模型")  # 改名
        self.llm_load_unload_button.clicked.connect(self.llm_toggle_model)
        llm_button_layout.addWidget(self.llm_load_unload_button)

        self.llm_clear_button = QPushButton("清空上下文")  # 改名
        self.llm_clear_button.clicked.connect(self.llm_clear_history_action)
        self.llm_clear_button.setEnabled(False)
        llm_button_layout.addWidget(self.llm_clear_button)
        
        self.llm_refresh_button = QPushButton("刷新模型")  # 改名
        self.llm_refresh_button.clicked.connect(self.llm_refresh_model_list)
        llm_button_layout.addWidget(self.llm_refresh_button)
        
        self.llm_download_button = QPushButton("下载模型")  # 改名
        llm_button_layout.addWidget(self.llm_download_button)

        # 控制台显示部分
        self.llm_console_display = QTextEdit()  # 改名
        self.llm_console_display.setReadOnly(True)
        self.llm_console_display.setStyleSheet("background-color: lightgray;")
        llm_layout.addWidget(self.llm_console_display, 3)

        # 用户输入部分
        llm_input_layout = QHBoxLayout()  # 改名
        llm_layout.addLayout(llm_input_layout)

        self.llm_user_input = QLineEdit()  # 改名
        self.llm_user_input.textChanged.connect(self.llm_update_send_button)
        llm_input_layout.addWidget(self.llm_user_input)

        self.llm_send_button = QPushButton("发送")  # 改名
        self.llm_send_button.clicked.connect(self.llm_send_message)
        self.llm_send_button.setEnabled(False)
        llm_input_layout.addWidget(self.llm_send_button)
        
        return llm_page_widget

    def ui_create_placeholder_page(self, text):
        """为正在开发的功能创建占位页面"""
        page_widget = QWidget()
        layout = QVBoxLayout(page_widget)
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        return page_widget

    def ui_create_t2i_page(self):
        """创建文生图页面"""
        page_widget = QWidget()
        layout = QVBoxLayout(page_widget)

        # 模型选择部分
        model_layout = QHBoxLayout()
        layout.addLayout(model_layout)

        model_label = QLabel("选择模型:")
        model_layout.addWidget(model_label)

        self.t2i_model_combo = QComboBox()
        # 使用动态获取的可用文生图模型列表
        if self.t2i_available_models:
            self.t2i_model_combo.addItems(self.t2i_available_models)
        else:
            self.t2i_model_combo.addItem("无可用模型")
            self.t2i_model_combo.setEnabled(False)
        model_layout.addWidget(self.t2i_model_combo)

        quant_label = QLabel("量化精度:")
        model_layout.addWidget(quant_label)

        self.t2i_quant_combo = QComboBox()
        self.t2i_quant_combo.addItems(T2I_QUANTIZATION_LIST)
        model_layout.addWidget(self.t2i_quant_combo)

        # 提示词输入部分
        prompt_layout = QVBoxLayout()
        layout.addLayout(prompt_layout)

        prompt_label = QLabel("提示词:")
        prompt_layout.addWidget(prompt_label)

        self.t2i_prompt_input = QTextEdit()
        self.t2i_prompt_input.setMaximumHeight(80)
        self.t2i_prompt_input.setPlaceholderText("输入图像生成提示词...")
        prompt_layout.addWidget(self.t2i_prompt_input)

        neg_prompt_label = QLabel("反向提示词:")
        prompt_layout.addWidget(neg_prompt_label)

        self.t2i_neg_prompt_input = QTextEdit()
        self.t2i_neg_prompt_input.setMaximumHeight(60)
        self.t2i_neg_prompt_input.setPlaceholderText("输入不希望出现的内容...")
        prompt_layout.addWidget(self.t2i_neg_prompt_input)

        # 参数设置部分
        params_layout = QHBoxLayout()
        layout.addLayout(params_layout)

        # 迭代次数
        steps_label = QLabel("迭代次数:")
        params_layout.addWidget(steps_label)
        self.t2i_steps_input = QSpinBox()
        self.t2i_steps_input.setRange(1, 100)
        self.t2i_steps_input.setValue(20)
        params_layout.addWidget(self.t2i_steps_input)

        # 种子
        seed_label = QLabel("种子:")
        params_layout.addWidget(seed_label)
        self.t2i_seed_input = QSpinBox()
        self.t2i_seed_input.setRange(-1, 999999)
        self.t2i_seed_input.setValue(-1)  # -1表示随机
        self.t2i_seed_input.setSpecialValueText("随机")
        params_layout.addWidget(self.t2i_seed_input)

        # 图片宽度 - 限制范围为256-512，步长为8
        width_label = QLabel("宽度:")
        params_layout.addWidget(width_label)
        self.t2i_width_input = QSpinBox()
        self.t2i_width_input.setRange(256, 512)  # 修改范围
        self.t2i_width_input.setValue(512)
        self.t2i_width_input.setSingleStep(8)  # 确保步长为8
        params_layout.addWidget(self.t2i_width_input)

        # 图片高度 - 限制范围为256-512，步长为8
        height_label = QLabel("高度:")
        params_layout.addWidget(height_label)
        self.t2i_height_input = QSpinBox()
        self.t2i_height_input.setRange(256, 512)  # 修改范围
        self.t2i_height_input.setValue(512)
        self.t2i_height_input.setSingleStep(8)  # 确保步长为8
        params_layout.addWidget(self.t2i_height_input)

        # 按钮部分
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        self.t2i_load_unload_button = QPushButton("加载模型")
        self.t2i_load_unload_button.clicked.connect(self.t2i_toggle_model)
        button_layout.addWidget(self.t2i_load_unload_button)

        self.t2i_generate_button = QPushButton("生成图像")
        self.t2i_generate_button.clicked.connect(self.t2i_generate_image)
        self.t2i_generate_button.setEnabled(False)
        button_layout.addWidget(self.t2i_generate_button)

        self.t2i_refresh_button = QPushButton("刷新模型")
        self.t2i_refresh_button.clicked.connect(self.t2i_refresh_model_list)
        button_layout.addWidget(self.t2i_refresh_button)

        self.t2i_download_button = QPushButton("下载模型")
        button_layout.addWidget(self.t2i_download_button)

        # 下方区域：左侧图片预览，右侧控制台
        bottom_layout = QHBoxLayout()
        layout.addLayout(bottom_layout)

        # 左侧图片预览区
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_label = QLabel("图片预览:")
        preview_layout.addWidget(preview_label)

        self.t2i_preview_area = QScrollArea()
        self.t2i_preview_area.setWidgetResizable(True)
        self.t2i_preview_area.setMinimumHeight(300)
        
        # 创建预览容器
        self.t2i_preview_container = QWidget()
        self.t2i_preview_layout = QVBoxLayout(self.t2i_preview_container)
        self.t2i_preview_area.setWidget(self.t2i_preview_container)
        
        preview_layout.addWidget(self.t2i_preview_area)
        bottom_layout.addWidget(preview_widget, 1)

        # 右侧控制台
        console_widget = QWidget()
        console_layout = QVBoxLayout(console_widget)
        console_label = QLabel("控制台:")
        console_layout.addWidget(console_label)

        self.t2i_console_display = QTextEdit()
        self.t2i_console_display.setReadOnly(True)
        self.t2i_console_display.setStyleSheet("background-color: lightgray;")
        self.t2i_console_display.setMinimumHeight(300)
        console_layout.addWidget(self.t2i_console_display)
        bottom_layout.addWidget(console_widget, 1)

        return page_widget

    def llm_toggle_model(self):
        # 根据当前状态决定是加载还是卸载模型
        if self.llm_manager.llm_pipe is None:
            # 如果模型未加载，则加载模型
            self.llm_load_model_action()
        else:
            # 如果模型已加载，则卸载模型
            self.llm_unload_model_action()

    def ui_update_llm_console(self, msg):  # 改名
        self.llm_console_display.append(msg)
        self.llm_console_display.verticalScrollBar().setValue(
            self.llm_console_display.verticalScrollBar().maximum()
        )
        QApplication.processEvents()

    def ui_update_llm_chat(self, msg):  # 改名
        self.llm_chat_display.insertPlainText(msg)

    def ui_update_llm_state(self, model_loaded):  # 改名
        self.llm_load_unload_button.setText("卸载模型" if model_loaded else "加载模型")
        self.llm_load_unload_button.setEnabled(True)  # 操作完成后启用按钮
        self.llm_clear_button.setEnabled(model_loaded)
        self.llm_user_input.setEnabled(model_loaded)
        self.llm_update_send_button()

    def ui_console_callback(self, msg):
        self.llm_console_signal.emit(msg)

    def ui_chat_callback(self, msg):
        self.llm_chat_signal.emit(msg)

    def chat_callback(self, msg):
        """为向后兼容保留的方法，实际调用新的方法"""
        self.ui_chat_callback(msg)

    def console_callback(self, msg):
        """为向后兼容保留的方法，实际调用新的方法"""
        self.ui_console_callback(msg)

    def llm_update_send_button(self):
        # 更新发送按钮的状态
        is_enabled = bool(self.llm_user_input.text().strip()) and self.llm_manager.llm_pipe is not None
        self.llm_send_button.setEnabled(is_enabled)
        self.llm_user_input.setEnabled(self.llm_manager.llm_pipe is not None)
        self.llm_clear_button.setEnabled(self.llm_manager.llm_pipe is not None)

    def llm_load_model_action(self):
        # 首先禁用按钮，防止重复点击
        self.llm_load_unload_button.setEnabled(False)
        self.llm_load_unload_button.setText("正在加载...")
        self.llm_console_signal.emit("开始加载LLM模型......\n")  # 修复信号名
        
        # 设置一个标志来控制加载状态指示器
        self.llm_loading_active = True
        
        # 实际加载模型的线程
        def load():
            success = False
            try:
                selected_model = self.llm_model_combo.currentText()
                selected_quant = self.llm_quant_combo.currentText()
                selected_device = self.llm_device_combo.currentText()
                
                # 使用修改后的回调函数
                success = self.llm_manager.llm_load_model(
                    selected_model, 
                    selected_quant, 
                    selected_device, 
                    lambda msg: self.llm_console_signal.emit(msg)  # 修复信号名
                )
            finally:
                # 无论成功与否，都停止加载状态指示器
                self.llm_loading_active = False
                # 在主线程中更新UI状态
                self.llm_update_ui_signal.emit(success)
        
        threading.Thread(target=load, daemon=True).start()

    def llm_unload_model_action(self):
        # 禁用按钮并更新文本
        self.llm_load_unload_button.setEnabled(False)
        self.llm_load_unload_button.setText("正在卸载...")
        
        # 卸载模型的逻辑
        self.llm_manager.llm_unload_model(self.ui_console_callback)
        
        # 更新UI状态
        self.llm_update_ui_signal.emit(False)  # 传递False表示模型已卸载
        
    def llm_clear_history_action(self):
        # 清空聊天历史记录的逻辑
        self.llm_manager.llm_clear_history()
        self.ui_console_callback("LLM上下文已清空\n\n")
        self.llm_update_send_button()

    def llm_send_message(self):
        # 发送消息的逻辑
        def send():
            user_input = self.llm_user_input.text()
            self.ui_chat_callback(f"\n\n用户: \n{user_input}\n")
            self.llm_user_input.clear()
            self.llm_send_button.setEnabled(False)
            self.ui_console_callback("LLM消息成功发送，等待输出中......\n")
            try:
                selected_model = self.llm_model_combo.currentText()
                prompt = self.llm_manager.llm_build_prompt(user_input, selected_model)
                self.ui_chat_callback("\n助手: \n")
                result = self.llm_manager.llm_generate_reply(prompt, self)
                perf_metrics = result.perf_metrics
                self.ui_console_callback(f"LLM已成功输出，速度为 {perf_metrics.get_throughput().mean:.2f} tokens/s\n\n")
                self.llm_manager.llm_append_history(user_input, result)
            except Exception as e:
                self.ui_chat_callback(f"助手: 无法生成回复，错误: {str(e)}\n\n")
            finally:
                self.llm_update_send_button()

        threading.Thread(target=send, daemon=True).start()

    def llm_refresh_model_list(self):
        """刷新LLM模型列表"""
        # 重新获取可用模型
        self.llm_available_models = self.ui_get_available_llm_models()  # 改名
        
        # 清空当前下拉框
        self.llm_model_combo.clear()
        
        # 重新填充下拉框
        if self.llm_available_models:
            self.llm_model_combo.addItems(self.llm_available_models)
            self.llm_model_combo.setEnabled(True)
            self.ui_console_callback(f"[LLM] 模型列表已刷新，找到 {len(self.llm_available_models)} 个可用模型\n")
            logging.info(f"[LLM] 刷新模型列表完成，可用模型: {self.llm_available_models}")
        else:
            self.llm_model_combo.addItem("无可用模型")
            self.llm_model_combo.setEnabled(False)
            self.ui_console_callback("[LLM] 未找到任何可用模型，请检查model文件夹\n")
            logging.warning("[LLM] 刷新后未找到任何可用模型")

    # 文生图相关方法
    def t2i_toggle_model(self):
        """切换文生图模型加载/卸载"""
        if self.t2i_manager.t2i_pipe is None:
            self.t2i_load_model_action()
        else:
            self.t2i_unload_model_action()

    def t2i_load_model_action(self):
        """加载文生图模型"""
        # 首先禁用按钮，防止重复点击
        self.t2i_load_unload_button.setEnabled(False)
        self.t2i_load_unload_button.setText("正在加载...")
        self.t2i_console_signal.emit("开始加载文生图模型......\n")
        
        # 设置一个标志来控制加载状态指示器
        self.t2i_loading_active = True
        
        # 实际加载模型的线程
        def load():
            success = False
            try:
                selected_model = self.t2i_model_combo.currentText()
                selected_quant = self.t2i_quant_combo.currentText()
                
                success = self.t2i_manager.t2i_load_model(
                    selected_model, 
                    selected_quant, 
                    lambda msg: self.t2i_console_signal.emit(msg)
                )
            finally:
                # 无论成功与否，都停止加载状态指示器
                self.t2i_loading_active = False
                # 在主线程中更新UI状态
                self.t2i_update_ui_signal.emit(success)
        
        threading.Thread(target=load, daemon=True).start()

    def t2i_unload_model_action(self):
        """卸载文生图模型"""
        self.t2i_load_unload_button.setEnabled(False)
        self.t2i_load_unload_button.setText("正在卸载...")
        
        self.t2i_manager.t2i_unload_model(
            lambda msg: self.t2i_console_signal.emit(msg)
        )
        
        self.t2i_update_ui_signal.emit(False)

    def t2i_generate_image(self):
        """生成图像"""
        if not self.t2i_manager.t2i_pipe:
            self.t2i_console_signal.emit("请先加载模型\n")
            return
            
        self.t2i_generate_button.setEnabled(False)
        self.t2i_generate_button.setText("正在生成...")
        
        prompt = self.t2i_prompt_input.toPlainText().strip()
        neg_prompt = self.t2i_neg_prompt_input.toPlainText().strip()
        steps = self.t2i_steps_input.value()
        seed = self.t2i_seed_input.value() if self.t2i_seed_input.value() >= 0 else None
        width = self.t2i_width_input.value()
        height = self.t2i_height_input.value()
        
        if not prompt:
            self.t2i_console_signal.emit("请输入提示词\n")
            self.t2i_generate_button.setEnabled(True)
            self.t2i_generate_button.setText("生成图像")
            return
        
        # 初始化步数显示
        self.t2i_progress_signal.emit(0, steps)
        
        # 生成图像的线程
        def generate():
            try:
                # 定义进度回调函数
                def progress_callback(step, total_steps):
                    self.t2i_progress_signal.emit(step, total_steps)
                
                result = self.t2i_manager.t2i_generate_image(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    num_images=1,
                    seed=seed,
                    console_callback=lambda msg: self.t2i_console_signal.emit(msg),
                    progress_callback=progress_callback
                )
                
                # 完成后显示完成信息
                self.t2i_console_signal.emit("图像生成完成！\n")
                
                # 发送图像信号
                self.t2i_image_signal.emit(result)
                
            except Exception as e:
                self.t2i_console_signal.emit(f"图像生成失败: {str(e)}\n")
                logging.error(f"T2I生成错误: {str(e)}")
            finally:
                # 恢复按钮状态
                self.t2i_generate_button.setEnabled(True)
                self.t2i_generate_button.setText("生成图像")
        
        threading.Thread(target=generate, daemon=True).start()

    def t2i_refresh_model_list(self):
        """刷新文生图模型列表"""
        self.t2i_available_models = self.ui_get_available_t2i_models()
        
        self.t2i_model_combo.clear()
        
        if self.t2i_available_models:
            self.t2i_model_combo.addItems(self.t2i_available_models)
            self.t2i_model_combo.setEnabled(True)
            self.t2i_console_signal.emit(f"[T2I] 文生图模型列表已刷新，找到 {len(self.t2i_available_models)} 个可用模型\n")
            logging.info(f"[T2I] 刷新模型列表完成，可用模型: {self.t2i_available_models}")
        else:
            self.t2i_model_combo.addItem("无可用模型")
            self.t2i_model_combo.setEnabled(False)
            self.t2i_console_signal.emit("[T2I] 未找到任何可用文生图模型，请检查model文件夹\n")
            logging.warning("[T2I] 刷新后未找到任何可用模型")

    # 文生图UI更新方法
    def ui_update_t2i_console(self, msg):
        """更新文生图控制台"""
        self.t2i_console_display.append(msg)
        self.t2i_console_display.verticalScrollBar().setValue(
            self.t2i_console_display.verticalScrollBar().maximum()
        )
        QApplication.processEvents()

    def ui_update_t2i_state(self, model_loaded):
        """更新文生图UI状态"""
        self.t2i_load_unload_button.setText("卸载模型" if model_loaded else "加载模型")
        self.t2i_load_unload_button.setEnabled(True)
        self.t2i_generate_button.setEnabled(model_loaded)

    def ui_update_t2i_images(self, image_paths):
        """更新图片预览区域"""
        # 清除之前的图片
        for i in reversed(range(self.t2i_preview_layout.count())):
            child = self.t2i_preview_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # 添加新图片
        for image_path in image_paths:
            if os.path.exists(image_path):
                label = QLabel()
                pixmap = QPixmap(image_path)
                # 缩放图片以适应预览区域
                scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
                label.setAlignment(Qt.AlignCenter)
                self.t2i_preview_layout.addWidget(label)
                
                # 添加图片路径标签
                path_label = QLabel(f"保存路径: {image_path}")
                path_label.setWordWrap(True)
                path_label.setStyleSheet("color: gray; font-size: 10px;")
                self.t2i_preview_layout.addWidget(path_label)

    def ui_update_t2i_progress(self, current_step, total_steps):
        """更新文生图进度显示 - 仅显示步数"""
        self.t2i_current_step = current_step
        self.t2i_total_steps = total_steps
        
        if total_steps > 0:
            # 清除之前的进度行（如果存在）
            text = self.t2i_console_display.toPlainText()
            lines = text.split('\n')
            
            # 移除之前的进度行
            while lines and (lines[-1].startswith('生成进度:') or lines[-1].startswith('当前步数:')):
                lines.pop()
            
            # 重新设置文本
            self.t2i_console_display.setPlainText('\n'.join(lines))
            
            # 添加新的步数显示
            step_text = f"当前步数: {current_step}/{total_steps}"
            self.t2i_console_display.append(step_text)
            
            # 滚动到底部
            self.t2i_console_display.verticalScrollBar().setValue(
                self.t2i_console_display.verticalScrollBar().maximum()
            )
            QApplication.processEvents()

    def ui_close_event_handler(self, event):
        # 在程序关闭前卸载模型并释放资源
        try:
            # 设置标志，终止所有正在运行的线程
            self.llm_loading_active = False
            self.t2i_loading_active = False
            
            # 如果LLM模型已加载，则卸载模型
            if self.llm_manager.llm_pipe is not None:
                self.ui_console_callback("程序正在关闭，卸载LLM模型并释放资源......\n")
                self.llm_manager.llm_unload_model(self.ui_console_callback)
            
            # 如果文生图模型已加载，则卸载模型
            if self.t2i_manager.t2i_pipe is not None:
                self.ui_console_callback("程序正在关闭，卸载文生图模型并释放资源......\n")
                self.t2i_manager.t2i_unload_model(self.ui_console_callback)
                
            self.ui_console_callback("资源已释放，程序已安全退出。\n")
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

    def closeEvent(self, event):
        self.ui_close_event_handler(event)