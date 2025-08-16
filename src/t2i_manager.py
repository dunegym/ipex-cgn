import logging
import os
import multiprocessing as mp
from t2i_worker import t2i_worker_process

class T2IManager:
    """管理T2I功能的类，通过多进程负责模型加载、卸载、图片生成等"""
    def __init__(self):
        """
        初始化T2I管理器
        """
        self.t2i_process = None
        self.t2i_input_queue = None
        self.t2i_output_queue = None

    def start_worker(self):
        """启动T2I工作进程"""
        if self.t2i_process and self.t2i_process.is_alive():
            return # Worker already running
        
        device = "HETERO:GPU,CPU" # T2I worker hardcodes this device
        logging.info(f"Starting T2I worker process on device: {device}")
        self.t2i_input_queue = mp.Queue()
        self.t2i_output_queue = mp.Queue()
        self.t2i_process = mp.Process(
            target=t2i_worker_process,
            args=(self.t2i_input_queue, self.t2i_output_queue)
        )
        self.t2i_process.start()

    def stop_worker(self):
        """停止T2I工作进程"""
        if self.t2i_process and self.t2i_process.is_alive():
            logging.info("Stopping T2I worker process.")
            self.t2i_input_queue.put(None) # Sentinel to stop the worker
            self.t2i_process.join(timeout=5)
            if self.t2i_process.is_alive():
                self.t2i_process.terminate()
        self.t2i_process = None
        self.t2i_input_queue = None
        self.t2i_output_queue = None

    def t2i_load_model(self, model_name, quant, console_callback=None):
        """
        向工作进程发送加载T2I模型的指令
        """
        self.stop_worker() # Stop any existing worker
        # T2I worker hardcodes device to "HETERO:GPU,CPU", so we don't need to pass device parameter
        self.start_worker() # Start a new one with the hardcoded device

        task = {
            'command': 'load',
            'model_name': model_name,
            'quant': quant
        }
        self.t2i_input_queue.put(task)
        logging.info(f"Sent 'load' command to T2I worker for model: {model_name} ({quant})")

    def t2i_unload_model(self, console_callback=None):
        """
        向工作进程发送卸载T2I模型的指令
        """
        if self.t2i_process and self.t2i_process.is_alive():
            logging.info("Sent 'unload' command to T2I worker.")
            task = {'command': 'unload'}
            self.t2i_input_queue.put(task)
        else:
            # 如果进程不存在，直接在主进程中清理
            if console_callback:
                console_callback("T2I模型已成功卸载！\n\n")
            logging.info("T2I model already unloaded (no worker process).")

    def t2i_generate_image(self, prompt, negative_prompt="", steps=20, seed=-1, width=512, height=512, guidance_scale=7.5, num_images_per_prompt=1):
        """
        向工作进程发送生成图片的指令
        """
        if not (self.t2i_process and self.t2i_process.is_alive()):
            err = RuntimeError("T2I worker process is not running.")
            logging.error(f"T2I错误: {err}")
            # 将错误放入队列，以便UI可以处理它
            self.t2i_output_queue.put({'status': 'error', 'message': str(err)})
            return

        task = {
            'command': 'generate',
            'params': {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'num_inference_steps': steps,
                'seed': seed,
                'width': width,
                'height': height,
                'guidance_scale': guidance_scale,
                'num_images': num_images_per_prompt
            }
        }
        self.t2i_input_queue.put(task)
        logging.info("Sent 'generate' command to T2I worker.")