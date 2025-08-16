import time
import logging
from transformers import AutoTokenizer
import openvino_genai as ov_genai
import openvino as ov
from PIL import Image
import os
import traceback
import numpy as np

# --- Absolute Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# --- Worker Process Core Logic ---

class VLMWorker:
    """处理VLM相关任务的工作类"""
    def __init__(self, device):
        self.pipe = None
        self.tokenizer = None
        self.device = device
        self.model_name = None
        self.quant = None

    def load_model(self, model_name, quant, output_queue):
        """加载VLM模型"""
        self.model_name = model_name
        self.quant = quant
        model_dir = os.path.join(project_root, "model", model_name, quant)
        logging.info(f"Worker: Loading VLM model {model_dir} on {self.device}")
        try:
            output_queue.put({'status': 'progress', 'data': 'Loading VLM model......\n'})
            start_time = time.time()
            
            output_queue.put({'status': 'progress', 'data': f"正在准备VLM模型目录: {model_dir}...\n"})
            output_queue.put({'status': 'progress', 'data': "开始加载VLM模型...\n"})
            self.pipe = ov_genai.VLMPipeline(model_dir, self.device)
            output_queue.put({'status': 'progress', 'data': "VLM模型加载完成。\n"})
            output_queue.put({'status': 'progress', 'data': "正在加载分词器...\n"})
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
            
            load_time = time.time() - start_time
            output_queue.put({'status': 'load_success', 'load_time': load_time})
            logging.info(f"Worker: VLM model loaded successfully in {load_time:.2f}s")
        except Exception as e:
            self.pipe = None
            self.tokenizer = None
            error_info = f"VLM模型加载失败: {str(e)}\n{traceback.format_exc()}"
            output_queue.put({'status': 'error', 'message': error_info})
            logging.error(error_info)

    def unload_model(self, output_queue):
        """卸载VLM模型"""
        self.pipe = None
        self.tokenizer = None
        self.model_name = None
        self.quant = None
        output_queue.put({'status': 'unload_success'})
        logging.info("Worker: VLM model unloaded.")

    def generate(self, image_path, prompt, max_new_tokens, output_queue):
        """生成图像描述"""
        if not self.pipe:
            output_queue.put({'status': 'error', 'message': 'VLM model not loaded.'})
            return

        try:
            # 打开并处理图像
            image = Image.open(image_path)
            image_data = np.array(image)
            image_tensor = ov.Tensor(image_data)
            
            output_queue.put({'status': 'progress', 'data': "开始生成图像描述...\n"})
            start_time = time.time()
            
            # 定义一个streamer，它将token发送到输出队列
            class QueueStreamer:
                def __init__(self, queue):
                    self.queue = queue
                def __call__(self, token_text: str):
                    self.queue.put({'status': 'vlm_chunk', 'data': token_text})
                    return False # 返回False以继续生成
                def end(self):
                    self.queue.put({'status': 'vlm_done'})

            streamer = QueueStreamer(output_queue)
            
            # 生成描述
            result = self.pipe.generate(
                prompt,
                image=image_tensor,
                max_new_tokens=max_new_tokens,
                streamer=streamer
            )
            
            # 确保在生成结束后发送完成信号
            streamer.end()
            
            generation_time = time.time() - start_time
            
            # 发送结果
            output_queue.put({'status': 'generate_success', 'text': result.texts[0], 'generation_time': generation_time})
            logging.info(f"VLM generation finished for image: {image_path} in {generation_time:.2f}s")

        except Exception as e:
            error_info = f"VLM推理失败: {str(e)}\n{traceback.format_exc()}"
            output_queue.put({'status': 'error', 'message': error_info})
            logging.error(error_info)


def vlm_worker_process(input_queue, output_queue, device):
    """VLM工作进程的主函数"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    worker = VLMWorker(device)
    logging.info(f"VLM Worker process started on device {device}.")
    
    while True:
        try:
            task = input_queue.get()
            if task is None:
                logging.info("VLM Worker process shutting down.")
                break

            command = task.get('command')
            if command == 'load':
                worker.load_model(task['model_name'], task['quant'], output_queue)
            elif command == 'unload':
                worker.unload_model(output_queue)
            elif command == 'generate':
                worker.generate(task['image_path'], task['prompt'], task['max_new_tokens'], output_queue)
            else:
                logging.warning(f"Unknown command: {command}")
        except Exception as e:
            error_info = f"VLM Worker process error: {str(e)}\n{traceback.format_exc()}"
            output_queue.put({'status': 'error', 'message': error_info})
            logging.error(error_info)
