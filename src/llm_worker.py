import time
import logging
from transformers import AutoTokenizer
import openvino_genai as ov_genai
import traceback
import os

# --- Absolute Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

class LLMWorker:
    """处理LLM相关任务的工作类"""
    def __init__(self, device):
        self.pipe = None
        self.tokenizer = None
        self.device = device
        self.model_name = None
        self.quant = None

    def load_model(self, model_name, quant, output_queue):
        """加载LLM模型"""
        self.model_name = model_name
        self.quant = quant
        model_dir = os.path.join(project_root, "model", model_name, quant)
        logging.info(f"Worker: Loading LLM model {model_dir} on {self.device}")
        try:
            output_queue.put({'status': 'progress', 'data': 'Loading LLM model......\n'})
            start_time = time.time()
            
            output_queue.put({'status': 'progress', 'data': f"正在准备LLM模型目录: {model_dir}...\n"})
            self.pipe = ov_genai.LLMPipeline(model_dir, self.device)
            
            output_queue.put({'status': 'progress', 'data': "LLM模型加载完成，正在加载分词器...\n"})
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
            
            load_time = time.time() - start_time
            output_queue.put({'status': 'load_success', 'load_time': load_time})
            logging.info(f"Worker: LLM model loaded successfully in {load_time:.2f}s")
        except Exception as e:
            self.pipe = None
            self.tokenizer = None
            error_info = f"LLM模型加载失败: {str(e)}\n{traceback.format_exc()}"
            output_queue.put({'status': 'error', 'message': error_info})
            logging.error(error_info)

    def unload_model(self, output_queue):
        """卸载LLM模型"""
        self.pipe = None
        self.tokenizer = None
        self.model_name = None
        self.quant = None
        output_queue.put({'status': 'unload_success'})
        logging.info("Worker: LLM model unloaded.")

    def generate(self, prompt, max_new_tokens, output_queue):
        """生成文本"""
        if not self.pipe:
            output_queue.put({'status': 'error', 'message': 'LLM model not loaded.'})
            return

        try:
            # 定义一个streamer，它将token发送到输出队列
            class QueueStreamer:
                def __init__(self, queue):
                    self.queue = queue
                def __call__(self, token_text: str):
                    self.queue.put({'status': 'chat_chunk', 'data': token_text})
                    return False # 返回False以继续生成
                def end(self):
                    self.queue.put({'status': 'chat_done'})

            streamer = QueueStreamer(output_queue)
            
            result = self.pipe.generate(
                [prompt],
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                use_cache=True
            )
            
            # 确保在生成结束后发送完成信号
            streamer.end()
            
            perf_metrics = result.perf_metrics
            throughput_raw = perf_metrics.get_throughput().mean if hasattr(perf_metrics, 'get_throughput') else 'N/A'
            if isinstance(throughput_raw, float):
                throughput = round(throughput_raw, 2)
            else:
                throughput = throughput_raw
            logging.info(f"LLM generation finished. Throughput: {throughput} tokens/s")
            output_queue.put({'status': 'generate_success', 'throughput': throughput})

        except Exception as e:
            error_info = f"LLM推理失败: {str(e)}\n{traceback.format_exc()}"
            output_queue.put({'status': 'error', 'message': error_info})
            logging.error(error_info)

def llm_worker_process(input_queue, output_queue, device):
    """LLM工作进程的主函数"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    worker = LLMWorker(device)
    logging.info(f"LLM Worker process started on device {device}.")
    
    while True:
        try:
            task = input_queue.get()
            if task is None:
                logging.info("LLM Worker process shutting down.")
                break

            command = task.get('command')
            if command == 'load':
                worker.load_model(task['model_name'], task['quant'], output_queue)
            elif command == 'unload':
                worker.unload_model(output_queue)
            elif command == 'generate':
                worker.generate(task['prompt'], task['max_new_tokens'], output_queue)
            else:
                logging.warning(f"Unknown command: {command}")
        except Exception as e:
            error_info = f"LLM Worker process error: {str(e)}\n{traceback.format_exc()}"
            output_queue.put({'status': 'error', 'message': error_info})
            logging.error(error_info)
