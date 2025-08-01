import logging
from transformers import AutoTokenizer
import multiprocessing as mp
from process_worker import llm_worker_process, t2i_worker_process

class LLMChatManager:
    """管理LLM聊天功能的类，通过多进程负责模型加载、卸载、聊天历史管理等"""
    def __init__(self):
        """
        初始化LLM聊天管理器
        """
        self.llm_chat_history = []  # 聊天历史记录
        self.llm_tokenizer = None # Tokenizer需要在主进程中用于构建prompt
        self.llm_process = None
        self.llm_input_queue = None
        self.llm_output_queue = None

    def start_worker(self, device):
        """启动LLM工作进程"""
        if self.llm_process and self.llm_process.is_alive():
            return # Worker already running
        
        logging.info(f"Starting LLM worker process on device: {device}")
        self.llm_input_queue = mp.Queue()
        self.llm_output_queue = mp.Queue()
        self.llm_process = mp.Process(
            target=llm_worker_process,
            args=(self.llm_input_queue, self.llm_output_queue, device)
        )
        self.llm_process.start()

    def stop_worker(self):
        """停止LLM工作进程"""
        if self.llm_process and self.llm_process.is_alive():
            logging.info("Stopping LLM worker process.")
            self.llm_input_queue.put(None) # Sentinel to stop the worker
            self.llm_process.join(timeout=5)
            if self.llm_process.is_alive():
                self.llm_process.terminate()
        self.llm_process = None
        self.llm_input_queue = None
        self.llm_output_queue = None

    def llm_clear_history(self):
        """
        清空聊天历史记录
        """
        self.llm_chat_history = []

    def llm_load_model(self, model_name, quant, device, console_callback=None):
        """
        向工作进程发送加载LLM模型的指令
        """
        self.stop_worker() # Stop any existing worker
        self.start_worker(device) # Start a new one with the correct device

        model_dir = f"model/{model_name}/{quant}"
        try:
            # Tokenizer可以在主进程中预加载，因为它通常很快
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
            logging.info("Tokenizer loaded in main process.")
        except Exception as e:
            self.llm_handle_exception(e, console_callback, prefix="LLM Tokenizer加载失败")
            # 即使tokenizer失败，也继续尝试加载模型，worker会报告更详细的错误
        
        task = {
            'command': 'load',
            'model_name': model_name,
            'quant': quant
        }
        self.llm_input_queue.put(task)
        logging.info(f"Sent 'load' command to LLM worker for model: {model_name} ({quant})")

    def llm_unload_model(self, console_callback=None):
        """
        向工作进程发送卸载LLM模型的指令
        """
        if self.llm_process and self.llm_process.is_alive():
            logging.info("Sent 'unload' command to LLM worker.")
            task = {'command': 'unload'}
            self.llm_input_queue.put(task)
        else:
            # 如果进程不存在，直接在主进程中清理
            self.llm_tokenizer = None
            if console_callback:
                console_callback("LLM模型已成功卸载！\n\n")
            logging.info("LLM model already unloaded (no worker process).")

    def llm_build_prompt(self, user_input, model_name):
        """
        构建聊天提示词
        """
        if not self.llm_tokenizer:
            self.llm_handle_exception(Exception("Tokenizer not loaded"), prefix="LLM Prompt 构建失败")
            return f"<|user|>\n{user_input}\n<|assistant|>\n"
        
        try:
            history = self.llm_chat_history[-6:] # 限制历史记录长度为6
            messages = [{"role": "system", "content": "You are a helpful assistant. "}] + history + [{"role": "user", "content": user_input}]
            return self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            self.llm_handle_exception(e, prefix="LLM Prompt 构建失败")
            return f"<|user|>\n{user_input}\n<|assistant|>\n"

    def llm_generate_reply(self, prompt, max_new_tokens=2048):
        """
        向工作进程发送生成回复的指令
        """
        if not (self.llm_process and self.llm_process.is_alive()):
            err = RuntimeError("LLM worker process is not running.")
            self.llm_handle_exception(err)
            # 将错误放入队列，以便UI可以处理它
            self.llm_output_queue.put({'status': 'error', 'message': str(err)})
            return

        task = {
            'command': 'generate',
            'prompt': prompt,
            'max_new_tokens': max_new_tokens
        }
        self.llm_input_queue.put(task)
        logging.info("Sent 'generate' command to LLM worker.")

    def llm_append_history(self, user_input, assistant_output):
        """
        将用户输入和助手输出添加到聊天历史记录中
        :param user_input: 用户输入
        :param assistant_output: 助手输出
        """
        self.llm_chat_history.append({"role": "user", "content": user_input})
        self.llm_chat_history.append({"role": "assistant", "content": str(assistant_output)})
        if len(self.llm_chat_history) > 6:
            self.llm_chat_history = self.llm_chat_history[-6:]  # 保持历史记录长度不超过6

    def llm_handle_exception(self, e, console_callback=None, prefix="LLM错误"):
        """
        处理异常并记录日志
        :param e: 异常对象
        :param console_callback: 控制台输出回调函数
        :param prefix: 错误信息前缀
        """
        msg = f"{prefix}: {str(e)}\n\n"
        logging.error(msg)
        if console_callback:
            console_callback(msg)

class T2IManager:
    """管理文生图功能的类，通过多进程负责模型加载、卸载、图像生成等"""
    def __init__(self):
        """
        初始化文生图管理器
        """
        self.t2i_process = None
        self.t2i_input_queue = None
        self.t2i_output_queue = None
        self.t2i_model_name = None
        self.t2i_quant = None

    def start_worker(self):
        """启动T2I工作进程"""
        if self.t2i_process and self.t2i_process.is_alive():
            return # Worker already running
        
        logging.info("Starting T2I worker process.")
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
            self.t2i_input_queue.put(None) # Sentinel
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
        self.stop_worker()
        self.start_worker()
        
        self.t2i_model_name = model_name
        self.t2i_quant = quant
        
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
            self.t2i_model_name = None
            self.t2i_quant = None
            if console_callback:
                console_callback("T2I模型已成功卸载！\n\n")
            logging.info("T2I model already unloaded (no worker process).")

    def t2i_generate_image(self, prompt, negative_prompt="", width=512, height=512,
                           num_inference_steps=20, num_images=1, seed=None):
        """
        向工作进程发送生成图像的指令
        """
        if not (self.t2i_process and self.t2i_process.is_alive()):
            err = RuntimeError("T2I worker process is not running.")
            self.t2i_handle_exception(err)
            self.t2i_output_queue.put({'status': 'error', 'message': str(err)})
            return

        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "num_images": num_images,
            "seed": seed
        }
        task = {'command': 'generate', 'params': params}
        self.t2i_input_queue.put(task)
        logging.info("Sent 'generate' command to T2I worker.")

    def t2i_handle_exception(self, e, console_callback=None, prefix="T2I错误"):
        """
        处理异常并记录日志
        :param e: 异常对象
        :param console_callback: 控制台输出回调函数
        :param prefix: 错误信息前缀
        """
        msg = f"{prefix}: {str(e)}\n\n"
        logging.error(msg)
        if console_callback:
            console_callback(msg)

    def __del__(self):
        """清理资源"""
        self.stop_worker()
