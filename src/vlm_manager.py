import logging
import os
import numpy as np
import openvino as ov
import openvino_genai
from PIL import Image
from transformers import AutoTokenizer
from vlm_worker import VLMWorker, vlm_worker_process
import multiprocessing as mp

# --- Absolute Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

class VLMManager:
    """管理VLM图生文功能的类，通过多进程负责模型加载、卸载、图像描述生成等"""
    def __init__(self):
        """
        初始化VLM管理器
        """
        self.vlm_process = None
        self.vlm_input_queue = None
        self.vlm_output_queue = None
        self.vlm_tokenizer = None
        
    def start_worker(self, device):
        """
        启动VLM工作进程
        """
        if self.vlm_process and self.vlm_process.is_alive():
            logging.warning("VLM worker is already running.")
            return
            
        self.vlm_input_queue = mp.Queue()
        self.vlm_output_queue = mp.Queue()
        
        self.vlm_process = mp.Process(target=vlm_worker_process, args=(self.vlm_input_queue, self.vlm_output_queue, device))
        self.vlm_process.start()
        logging.info(f"VLM worker process started with PID: {self.vlm_process.pid}")
        
    def stop_worker(self):
        """
        停止VLM工作进程
        """
        if self.vlm_process and self.vlm_process.is_alive():
            logging.info("Terminating VLM worker process...")
            self.vlm_process.terminate()
            self.vlm_process.join(timeout=5)
            if self.vlm_process.is_alive():
                logging.warning("VLM worker process did not terminate gracefully, killing it.")
                self.vlm_process.kill()
                self.vlm_process.join()
        self.vlm_process = None
        self.vlm_input_queue = None
        self.vlm_output_queue = None
        
    def vlm_clear_history(self):
        """
        清空历史记录（如果需要）
        """
        pass
        
    def vlm_load_model(self, model_name, quant, device, console_callback=None):
        """
        向工作进程发送加载VLM模型的指令
        """
        self.stop_worker() # Stop any existing worker
        self.start_worker(device) # Start a new one with the correct device

        model_dir = os.path.join(project_root, "model", model_name, quant)
        try:
            # Tokenizer可以在主进程中预加载，因为它通常很快
            self.vlm_tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
            logging.info("VLM Tokenizer loaded in main process.")
        except Exception as e:
            logging.error(f"VLM Tokenizer加载失败: {str(e)}")
            # 即使tokenizer失败，也继续尝试加载模型，worker会报告更详细的错误
        
        task = {
            'command': 'load',
            'model_name': model_name,
            'quant': quant
        }
        self.vlm_input_queue.put(task)
        logging.info(f"Sent 'load' command to VLM worker for model: {model_name} ({quant})")
        
    def vlm_unload_model(self, console_callback=None):
        """
        向工作进程发送卸载VLM模型的指令
        """
        if self.vlm_process and self.vlm_process.is_alive():
            logging.info("Sent 'unload' command to VLM worker.")
            task = {'command': 'unload'}
            self.vlm_input_queue.put(task)
        else:
            # 如果进程不存在，直接在主进程中清理
            self.vlm_tokenizer = None
            if console_callback:
                console_callback("VLM模型已成功卸载！\n\n")
            logging.info("VLM model already unloaded (no worker process).")
            
    def vlm_generate_description(self, image_path, prompt_text, max_new_tokens=1000, console_callback=None):
        """
        向工作进程发送生成图像描述的指令
        """
        if not self.vlm_process or not self.vlm_process.is_alive():
            if console_callback:
                console_callback("错误：VLM模型未加载\n\n")
            logging.error("VLM model is not loaded.")
            return
            
        # 使用tokenizer生成prompt
        try:
            if not self.vlm_tokenizer:
                raise Exception("VLM tokenizer is not loaded.")
                
            prompt = self.vlm_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}], 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            if console_callback:
                console_callback(f"Prompt生成失败: {str(e)}\n\n")
            logging.error(f"Prompt generation failed: {str(e)}")
            return
            
        task = {
            'command': 'generate',
            'image_path': image_path,
            'prompt': prompt,
            'max_new_tokens': max_new_tokens
        }
        self.vlm_input_queue.put(task)
        logging.info(f"Sent 'generate' command to VLM worker for image: {image_path}")