from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QComboBox, QTextEdit, QLineEdit, QWidget, QStackedWidget, 
                             QSpinBox, QScrollArea, QDialog, QTableWidget, QTableWidgetItem, 
                             QHeaderView)
from PyQt5.QtCore import QTimer, Qt, QUrl
from PyQt5.QtGui import QPixmap, QFont, QDesktopServices, QTextCursor
import logging
import os
from queue import Empty
from config import (LLM_MODEL_DICT, LLM_QUANTIZATION_LIST, LLM_DEVICE_LIST, 
                    T2I_MODEL_DICT, T2I_QUANTIZATION_LIST)
from manager import LLMChatManager, T2IManager

class ModelDownloadDialog(QDialog):
    """模型下载信息对话框"""
    def __init__(self, model_dict, model_type, parent=None):
        super().__init__(parent)
        self.model_dict = model_dict
        self.model_type = model_type
        self.ui_init_download_dialog()
    
    def ui_init_download_dialog(self):
        self.setWindowTitle(f"{self.model_type}模型下载")
        self.setGeometry(200, 200, 800, 500)
        self.setModal(True)
        layout = QVBoxLayout(self)
        title_label = QLabel(f"{self.model_type}模型下载信息")
        title_font = QFont(); title_font.setPointSize(14); title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        info_label = QLabel("请复制下载地址到浏览器或下载工具中下载模型文件")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: gray; margin: 10px;")
        layout.addWidget(info_label)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["模型名称", "下载地址", "操作"])
        self.table.setRowCount(len(self.model_dict))
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        for row, (model_name, download_url) in enumerate(self.model_dict.items()):
            self.table.setItem(row, 0, QTableWidgetItem(model_name))
            self.table.setItem(row, 1, QTableWidgetItem(download_url))
            open_button = QPushButton("打开")
            open_button.clicked.connect(lambda _, url=download_url: self.open_url(url))
            self.table.setCellWidget(row, 2, open_button)
        layout.addWidget(self.table)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        button_layout = QHBoxLayout(); button_layout.addStretch(); button_layout.addWidget(close_button); button_layout.addStretch()
        layout.addLayout(button_layout)
        self.table.verticalHeader().setVisible(False)

    def open_url(self, url_string):
        QDesktopServices.openUrl(QUrl(url_string))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.llm_manager = LLMChatManager()
        self.t2i_manager = T2IManager()
        self.llm_available_models = self.ui_get_available_llm_models()
        self.t2i_available_models = self.ui_get_available_t2i_models()

        # 状态标志
        self.llm_model_loaded = False
        self.t2i_model_loaded = False
        self.llm_is_busy = False
        self.t2i_is_busy = False
        self.t2i_current_image_num = 0
        
        self.ui_init()

        # 设置一个定时器来处理来自工作进程的队列消息
        self.queue_timer = QTimer(self)
        self.queue_timer.timeout.connect(self.process_queues)
        self.queue_timer.start(100) # 每100毫秒检查一次队列



    def ui_init(self):
        self.setWindowTitle("OpenVINO多功能AI平台")
        self.setGeometry(100, 100, 1600, 1200)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        font = self.font(); font.setPointSize(int(font.pointSize() * 1.3)); self.setFont(font)
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
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        self.text_to_text_page = self.ui_create_llm_page()
        self.text_to_image_page = self.ui_create_t2i_page()
        self.image_to_text_page = self.ui_create_placeholder_page("图生文功能开发中...")
        self.speech_rec_page = self.ui_create_placeholder_page("语音识别功能开发中...")
        self.stacked_widget.addWidget(self.text_to_text_page)
        self.stacked_widget.addWidget(self.text_to_image_page)
        self.stacked_widget.addWidget(self.image_to_text_page)
        self.stacked_widget.addWidget(self.speech_rec_page)
        self.text_to_text_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.text_to_image_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        self.image_to_text_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))
        self.speech_rec_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(3))
        self.stacked_widget.setCurrentIndex(0)

    def ui_create_llm_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("选择模型:"))
        self.llm_model_combo = QComboBox()
        if self.llm_available_models: self.llm_model_combo.addItems(self.llm_available_models)
        else: self.llm_model_combo.addItem("无可用模型"); self.llm_model_combo.setEnabled(False)
        model_layout.addWidget(self.llm_model_combo)
        model_layout.addWidget(QLabel("量化精度:"))
        self.llm_quant_combo = QComboBox(); self.llm_quant_combo.addItems(LLM_QUANTIZATION_LIST)
        model_layout.addWidget(self.llm_quant_combo)
        model_layout.addWidget(QLabel("选择设备:"))
        self.llm_device_combo = QComboBox(); self.llm_device_combo.addItems(LLM_DEVICE_LIST)
        model_layout.addWidget(self.llm_device_combo)
        layout.addLayout(model_layout)
        self.llm_chat_display = QTextEdit(); self.llm_chat_display.setReadOnly(True)
        layout.addWidget(self.llm_chat_display, 7)
        btn_layout = QHBoxLayout()
        self.llm_load_unload_button = QPushButton("加载模型"); self.llm_load_unload_button.clicked.connect(self.llm_toggle_model)
        btn_layout.addWidget(self.llm_load_unload_button)
        self.llm_clear_button = QPushButton("清空上下文"); self.llm_clear_button.clicked.connect(self.llm_clear_history_action); self.llm_clear_button.setEnabled(False)
        btn_layout.addWidget(self.llm_clear_button)
        self.llm_refresh_button = QPushButton("刷新模型"); self.llm_refresh_button.clicked.connect(self.llm_refresh_model_list)
        btn_layout.addWidget(self.llm_refresh_button)
        self.llm_download_button = QPushButton("下载模型"); self.llm_download_button.clicked.connect(self.llm_show_download_dialog)
        btn_layout.addWidget(self.llm_download_button)
        layout.addLayout(btn_layout)
        self.llm_console_display = QTextEdit(); self.llm_console_display.setReadOnly(True); self.llm_console_display.setStyleSheet("background-color: lightgray;")
        layout.addWidget(self.llm_console_display, 3)
        input_layout = QHBoxLayout()
        self.llm_user_input = QLineEdit(); self.llm_user_input.textChanged.connect(self.llm_update_send_button)
        input_layout.addWidget(self.llm_user_input)
        self.llm_send_button = QPushButton("发送"); self.llm_send_button.clicked.connect(self.llm_send_message); self.llm_send_button.setEnabled(False)
        input_layout.addWidget(self.llm_send_button)
        layout.addLayout(input_layout)
        return page

    def ui_create_t2i_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("选择模型:"))
        self.t2i_model_combo = QComboBox()
        if self.t2i_available_models: self.t2i_model_combo.addItems(self.t2i_available_models)
        else: self.t2i_model_combo.addItem("无可用模型"); self.t2i_model_combo.setEnabled(False)
        model_layout.addWidget(self.t2i_model_combo)
        model_layout.addWidget(QLabel("量化精度:"))
        self.t2i_quant_combo = QComboBox(); self.t2i_quant_combo.addItems(T2I_QUANTIZATION_LIST)
        model_layout.addWidget(self.t2i_quant_combo)
        layout.addLayout(model_layout)
        prompt_layout = QVBoxLayout()
        prompt_layout.addWidget(QLabel("提示词:"))
        self.t2i_prompt_input = QTextEdit(); self.t2i_prompt_input.setMaximumHeight(80); self.t2i_prompt_input.setPlaceholderText("输入图像生成提示词...")
        prompt_layout.addWidget(self.t2i_prompt_input)
        prompt_layout.addWidget(QLabel("反向提示词:"))
        self.t2i_neg_prompt_input = QTextEdit(); self.t2i_neg_prompt_input.setMaximumHeight(60); self.t2i_neg_prompt_input.setPlaceholderText("输入不希望出现的内容...")
        prompt_layout.addWidget(self.t2i_neg_prompt_input)
        layout.addLayout(prompt_layout)
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("迭代次数:")); self.t2i_steps_input = QSpinBox(); self.t2i_steps_input.setRange(1, 100); self.t2i_steps_input.setValue(20); params_layout.addWidget(self.t2i_steps_input)
        params_layout.addWidget(QLabel("种子:")); self.t2i_seed_input = QSpinBox(); self.t2i_seed_input.setRange(-1, 999999); self.t2i_seed_input.setValue(-1); self.t2i_seed_input.setSpecialValueText("随机"); params_layout.addWidget(self.t2i_seed_input)
        params_layout.addWidget(QLabel("宽度:")); self.t2i_width_input = QSpinBox(); self.t2i_width_input.setRange(256, 512); self.t2i_width_input.setValue(512); self.t2i_width_input.setSingleStep(8); params_layout.addWidget(self.t2i_width_input)
        params_layout.addWidget(QLabel("高度:")); self.t2i_height_input = QSpinBox(); self.t2i_height_input.setRange(256, 512); self.t2i_height_input.setValue(512); self.t2i_height_input.setSingleStep(8); params_layout.addWidget(self.t2i_height_input)
        params_layout.addWidget(QLabel("生图数量:")); self.t2i_num_images_input = QSpinBox(); self.t2i_num_images_input.setRange(1, 100); self.t2i_num_images_input.setValue(1); params_layout.addWidget(self.t2i_num_images_input)
        layout.addLayout(params_layout)
        btn_layout = QHBoxLayout()
        self.t2i_load_unload_button = QPushButton("加载模型"); self.t2i_load_unload_button.clicked.connect(self.t2i_toggle_model)
        btn_layout.addWidget(self.t2i_load_unload_button)
        self.t2i_generate_button = QPushButton("生成图像"); self.t2i_generate_button.clicked.connect(self.t2i_generate_image); self.t2i_generate_button.setEnabled(False)
        btn_layout.addWidget(self.t2i_generate_button)
        self.t2i_refresh_button = QPushButton("刷新模型"); self.t2i_refresh_button.clicked.connect(self.t2i_refresh_model_list)
        btn_layout.addWidget(self.t2i_refresh_button)
        self.t2i_download_button = QPushButton("下载模型"); self.t2i_download_button.clicked.connect(self.t2i_show_download_dialog)
        btn_layout.addWidget(self.t2i_download_button)
        layout.addLayout(btn_layout)
        bottom_layout = QHBoxLayout()
        preview_widget = QWidget(); preview_layout = QVBoxLayout(preview_widget); preview_layout.addWidget(QLabel("图片预览:"))
        self.t2i_preview_area = QScrollArea(); self.t2i_preview_area.setWidgetResizable(True); self.t2i_preview_area.setMinimumHeight(300)
        self.t2i_preview_container = QWidget(); self.t2i_preview_layout = QVBoxLayout(self.t2i_preview_container)
        self.t2i_preview_area.setWidget(self.t2i_preview_container)
        preview_layout.addWidget(self.t2i_preview_area)
        bottom_layout.addWidget(preview_widget, 2)
        console_widget = QWidget(); console_layout = QVBoxLayout(console_widget); console_layout.addWidget(QLabel("控制台:"))
        self.t2i_console_display = QTextEdit(); self.t2i_console_display.setReadOnly(True); self.t2i_console_display.setStyleSheet("background-color: lightgray;"); self.t2i_console_display.setMinimumHeight(300)
        console_layout.addWidget(self.t2i_console_display)
        bottom_layout.addWidget(console_widget, 3)
        layout.addLayout(bottom_layout)
        return page

    def ui_create_placeholder_page(self, text):
        page_widget = QWidget()
        layout = QVBoxLayout(page_widget)
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        return page_widget

    def process_queues(self):
        # 处理LLM队列
        if self.llm_manager.llm_output_queue:
            while not self.llm_manager.llm_output_queue.empty():
                try:
                    msg = self.llm_manager.llm_output_queue.get_nowait()
                    self.handle_llm_message(msg)
                except Empty:
                    break
        
        # 处理T2I队列
        if self.t2i_manager.t2i_output_queue:
            while not self.t2i_manager.t2i_output_queue.empty():
                try:
                    msg = self.t2i_manager.t2i_output_queue.get_nowait()
                    self.handle_t2i_message(msg)
                except Empty:
                    break

    def handle_llm_message(self, msg):
        status = msg.get('status')
        if status == 'progress':
            self.ui_update_llm_console(msg.get('data', ''))
        elif status == 'load_success':
            self.llm_model_loaded = True
            self.llm_is_busy = False
            self.ui_update_llm_state()
            self.ui_update_llm_console(f"LLM模型加载成功！耗时：{msg.get('load_time', 0):.2f} 秒\n\n")
        elif status == 'unload_success':
            self.llm_model_loaded = False
            self.llm_is_busy = False
            self.llm_manager.llm_tokenizer = None # 清理tokenizer
            self.ui_update_llm_state()
            self.ui_update_llm_console("LLM模型已成功卸载！\n\n")
        elif status == 'chat_chunk':
            self.llm_chat_display.insertPlainText(msg.get('data', ''))
        elif status == 'chat_done':
            self.llm_is_busy = False
            self.ui_update_llm_state()
        elif status == 'generate_success':
            self.llm_is_busy = False
            self.ui_update_llm_state()
            self.ui_update_llm_console(f"LLM生成完成, 速度: {msg.get('throughput', 'N/A')} tokens/s\n")
        elif status == 'error':
            self.llm_is_busy = False
            self.ui_update_llm_state()
            self.ui_update_llm_console(f"LLM错误: {msg.get('message', '未知错误')}\n")

    def handle_t2i_message(self, msg):
        status = msg.get('status')
        if status == 'progress':
            self.ui_update_t2i_console(msg.get('data', ''))
        elif status == 'load_success':
            self.t2i_model_loaded = True
            self.t2i_is_busy = False
            self.ui_update_t2i_state()
            self.ui_update_t2i_console(f"T2I模型加载成功！耗时：{msg.get('load_time', 0):.2f} 秒\n\n")
        elif status == 'unload_success':
            self.t2i_model_loaded = False
            self.t2i_is_busy = False
            self.ui_update_t2i_state()
            self.ui_update_t2i_console("T2I模型已成功卸载！\n\n")
        elif status == 'image_generated':
            self.ui_append_t2i_image(msg.get('path'))
        elif status == 't2i_progress_update':
            self.ui_update_t2i_progress(msg['step'], msg['total'], msg['image_num'], msg['total_images'])
        elif status == 'generate_success':
            self.t2i_is_busy = False
            self.ui_update_t2i_state()
            self.t2i_generate_button.setText("生成图像") # 恢复按钮文本
            self.ui_update_t2i_console("\n--- 所有图像生成完毕 ---\n\n")
        elif status == 'error':
            self.t2i_is_busy = False
            self.ui_update_t2i_state()
            self.ui_update_t2i_console(f"T2I错误: {msg.get('message', '未知错误')}\n")

    def llm_toggle_model(self):
        if self.llm_is_busy: return
        if not self.llm_model_loaded:
            self.llm_load_model_action()
        else:
            self.llm_unload_model_action()

    def llm_load_model_action(self):
        self.llm_is_busy = True
        self.ui_update_llm_state()
        selected_model = self.llm_model_combo.currentText()
        selected_quant = self.llm_quant_combo.currentText()
        selected_device = self.llm_device_combo.currentText()
        self.llm_manager.llm_load_model(selected_model, selected_quant, selected_device)

    def llm_unload_model_action(self):
        self.llm_is_busy = True
        self.ui_update_llm_state()
        self.llm_manager.llm_unload_model()
        
    def llm_clear_history_action(self):
        self.llm_manager.llm_clear_history()
        self.llm_chat_display.clear()
        self.ui_update_llm_console("LLM上下文已清空\n\n")

    def llm_send_message(self):
        if self.llm_is_busy: return
        user_input = self.llm_user_input.text().strip()
        if not user_input: return
        
        self.llm_is_busy = True
        self.ui_update_llm_state()
        self.llm_user_input.clear()
        self.llm_chat_display.append(f"\n\n<b>用户:</b>\n{user_input}\n")
        self.llm_chat_display.append("\n<b>助手:</b>\n")
        
        selected_model = self.llm_model_combo.currentText()
        prompt = self.llm_manager.llm_build_prompt(user_input, selected_model)
        self.llm_manager.llm_append_history(user_input, "") # 预先添加历史，回复由streamer填充
        self.llm_manager.llm_generate_reply(prompt)

    def llm_refresh_model_list(self):
        self.llm_available_models = self.ui_get_available_llm_models()
        self.llm_model_combo.clear()
        if self.llm_available_models:
            self.llm_model_combo.addItems(self.llm_available_models)
            self.llm_model_combo.setEnabled(True)
        else:
            self.llm_model_combo.addItem("无可用模型"); self.llm_model_combo.setEnabled(False)
        self.ui_update_llm_console("[LLM] 模型列表已刷新\n")

    def ui_update_llm_console(self, msg):
        self.llm_console_display.append(msg)
        self.llm_console_display.verticalScrollBar().setValue(self.llm_console_display.verticalScrollBar().maximum())

    def ui_update_llm_state(self):
        model_loaded = self.llm_model_loaded
        is_busy = self.llm_is_busy
        self.llm_load_unload_button.setText("卸载模型" if model_loaded else "加载模型")
        self.llm_load_unload_button.setEnabled(not is_busy)
        if is_busy and not model_loaded: self.llm_load_unload_button.setText("正在加载...")
        if is_busy and model_loaded: self.llm_load_unload_button.setText("正在操作...")
        
        self.llm_clear_button.setEnabled(model_loaded and not is_busy)
        self.llm_user_input.setEnabled(model_loaded and not is_busy)
        self.llm_send_button.setEnabled(model_loaded and not is_busy and bool(self.llm_user_input.text().strip()))

    def llm_update_send_button(self):
        self.ui_update_llm_state()

    def llm_show_download_dialog(self):
        dialog = ModelDownloadDialog(LLM_MODEL_DICT, "文生文", self)
        dialog.exec_()

    def t2i_toggle_model(self):
        if self.t2i_is_busy: return
        if not self.t2i_model_loaded:
            self.t2i_load_model_action()
        else:
            self.t2i_unload_model_action()

    def t2i_load_model_action(self):
        self.t2i_is_busy = True
        self.ui_update_t2i_state()
        selected_model = self.t2i_model_combo.currentText()
        selected_quant = self.t2i_quant_combo.currentText()
        self.t2i_manager.t2i_load_model(selected_model, selected_quant)

    def t2i_unload_model_action(self):
        self.t2i_is_busy = True
        self.ui_update_t2i_state()
        self.t2i_manager.t2i_unload_model()

    def t2i_generate_image(self):
        if self.t2i_is_busy or not self.t2i_model_loaded: return
        prompt = self.t2i_prompt_input.toPlainText().strip()
        if not prompt:
            self.ui_update_t2i_console("请输入提示词\n")
            return
            
        self.t2i_is_busy = True
        self.t2i_current_image_num = 0 # Reset for new generation sequence
        self.ui_update_t2i_state()
        self.ui_clear_t2i_preview()
        
        params = {
            "prompt": prompt,
            "negative_prompt": self.t2i_neg_prompt_input.toPlainText().strip(),
            "num_inference_steps": self.t2i_steps_input.value(),
            "seed": self.t2i_seed_input.value() if self.t2i_seed_input.value() >= 0 else None,
            "width": self.t2i_width_input.value(),
            "height": self.t2i_height_input.value(),
            "num_images": self.t2i_num_images_input.value()
        }
        self.t2i_manager.t2i_generate_image(**params)

    def t2i_refresh_model_list(self):
        self.t2i_available_models = self.ui_get_available_t2i_models()
        self.t2i_model_combo.clear()
        if self.t2i_available_models:
            self.t2i_model_combo.addItems(self.t2i_available_models)
            self.t2i_model_combo.setEnabled(True)
        else:
            self.t2i_model_combo.addItem("无可用模型"); self.t2i_model_combo.setEnabled(False)
        self.ui_update_t2i_console("[T2I] 模型列表已刷新\n")

    def ui_update_t2i_console(self, msg):
        self.t2i_console_display.append(msg)
        self.t2i_console_display.verticalScrollBar().setValue(self.t2i_console_display.verticalScrollBar().maximum())

    def ui_update_t2i_state(self):
        model_loaded = self.t2i_model_loaded
        is_busy = self.t2i_is_busy
        self.t2i_load_unload_button.setText("卸载模型" if model_loaded else "加载模型")
        self.t2i_load_unload_button.setEnabled(not is_busy)
        if is_busy and not model_loaded: self.t2i_load_unload_button.setText("正在加载...")
        if is_busy and model_loaded: self.t2i_load_unload_button.setText("正在操作...")
        
        self.t2i_generate_button.setEnabled(model_loaded and not is_busy)

    def ui_clear_t2i_preview(self):
        for i in reversed(range(self.t2i_preview_layout.count())):
            child = self.t2i_preview_layout.itemAt(i).widget()
            if child: child.setParent(None)

    def ui_append_t2i_image(self, image_path):
        try:
            image_label = QLabel()
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setToolTip(f"图像路径: {image_path}")
            self.t2i_preview_layout.addWidget(image_label)
        except Exception as e:
            logging.error(f"显示图像时出错: {str(e)}")

    def ui_update_t2i_progress(self, current_step, total_steps, image_num, total_images):
        if total_steps <= 0:
            return

        # --- UI-side Progress Bar Construction ---
        percentage = (current_step + 1) / total_steps
        self.t2i_generate_button.setText(f"生成中({int(percentage * 100)}%)")

        bar_length = 15
        filled_length = int(bar_length * percentage)
        bar = '█' * filled_length + ' ' * (bar_length - filled_length)
        
        bar_text = f"生成图像 {image_num}/{total_images}: {int(percentage * 100):3d}%|{bar}| {current_step + 1}/{total_steps}"

        cursor = self.t2i_console_display.textCursor()

        # Check if we are starting a new image's progress bar
        if image_num != self.t2i_current_image_num:
            self.t2i_current_image_num = image_num
            # Add a newline for separation and append the new bar
            self.t2i_console_display.append("")
            self.t2i_console_display.append(bar_text)
        else:
            # We are updating the progress for the current image.
            # Move cursor to the beginning of the last line
            cursor.movePosition(QTextCursor.End)
            cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.MoveAnchor)
            # Select the entire line
            cursor.movePosition(QTextCursor.EndOfLine, QTextCursor.KeepAnchor)
            # Replace it with the new progress text
            cursor.insertText(bar_text)
        
        self.t2i_console_display.setTextCursor(cursor)

    def t2i_show_download_dialog(self):
        dialog = ModelDownloadDialog(T2I_MODEL_DICT, "文生图", self)
        dialog.exec_()

    def closeEvent(self, event):
        self.llm_manager.stop_worker()
        self.t2i_manager.stop_worker()
        event.accept()

    def ui_get_available_llm_models(self):
        """检查model文件夹下实际存在的LLM模型"""
        available_models = []
        model_dir = os.path.join(os.path.dirname(__file__), 'model')
        if not os.path.exists(model_dir):
            logging.warning(f"[LLM] 模型目录不存在: {model_dir}")
            return available_models
        
        for model_name in LLM_MODEL_DICT.keys():
            if os.path.isdir(os.path.join(model_dir, model_name)):
                available_models.append(model_name)
                logging.info(f"[LLM] 发现可用LLM模型: {model_name}")
        
        if not available_models:
            logging.warning("[LLM] 未找到任何与配置匹配的可用LLM模型")
        return available_models
    
    def ui_get_available_t2i_models(self):
        """检查model文件夹下实际存在的T2I模型"""
        available_models = []
        model_dir = os.path.join(os.path.dirname(__file__), 'model')
        if not os.path.exists(model_dir):
            logging.warning(f"[T2I] 模型目录不存在: {model_dir}")
            return available_models

        for model_name in T2I_MODEL_DICT.keys():
            if os.path.isdir(os.path.join(model_dir, model_name)):
                available_models.append(model_name)
                logging.info(f"[T2I] 发现可用T2I模型: {model_name}")

        if not available_models:
            logging.warning("[T2I] 未找到任何与配置匹配的可用T2I模型")
        return available_models

    def run_automated_test(self):
        """自动化测试，用于验证tqdm进度条"""
        logging.info("Starting automated test...")
        # 1. 切换到T2I页面
        self.stacked_widget.setCurrentIndex(1)
        self.ui_update_t2i_console("自动化测试：已切换到文生图页面。\n")

        # 2. 加载模型
        self.t2i_load_model_action()
        self.ui_update_t2i_console("自动化测试：正在加载T2I模型...\n")

        # 3. 等待模型加载完成，然后生成图像
        def check_t2i_model_loaded():
            if self.t2i_model_loaded:
                self.ui_update_t2i_console("自动化测试：T2I模型加载成功，开始生成图像。\n")
                self.t2i_prompt_input.setPlainText("a beautiful cat")
                self.t2i_num_images_input.setValue(3) # Set to generate 3 images
                self.t2i_generate_image()
            else:
                # 如果模型仍在加载，则稍后再次检查
                QTimer.singleShot(500, check_t2i_model_loaded)
        
        # 启动检查
        QTimer.singleShot(1000, check_t2i_model_loaded)
