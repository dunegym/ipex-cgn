import time
import logging
from transformers import AutoTokenizer
import openvino_genai as ov_genai
from PIL import Image
import tqdm
import sys
import os
import random
import traceback

# --- Worker Process Core Logic ---

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
        model_dir = f"model/{model_name}/{quant}"
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
            throughput = perf_metrics.get_throughput().mean if hasattr(perf_metrics, 'get_throughput') else 'N/A'
            logging.info(f"LLM generation finished. Throughput: {throughput} tokens/s")
            output_queue.put({'status': 'generate_success', 'throughput': throughput})

        except Exception as e:
            error_info = f"LLM推理失败: {str(e)}\n{traceback.format_exc()}"
            output_queue.put({'status': 'error', 'message': error_info})
            logging.error(error_info)


class T2IWorker:
    """处理T2I相关任务的工作类"""
    def __init__(self):
        self.pipe = None
        self.model_name = None
        self.quant = None
        self.config = {
            "HETERO_CONFIG_FILE": os.path.join(os.getcwd(), "hetero_config.xml"),
            "PERFORMANCE_HINT": "THROUGHPUT"
        }

    def load_model(self, model_name, quant, output_queue):
        """加载T2I模型"""
        self.model_name = model_name
        self.quant = quant
        model_dir = f"model/{model_name}/{quant}"
        logging.info(f"Worker: Loading T2I model {model_dir}")
        try:
            output_queue.put({'status': 'progress', 'data': 'Loading T2I model......\n'})
            start_time = time.time()
            
            output_queue.put({'status': 'progress', 'data': f"正在准备T2I模型目录: {model_dir}...\n"})
            output_queue.put({'status': 'progress', 'data': f"正在加载T2I模型到HETERO:GPU,CPU设备...\n"})
            
            self.pipe = ov_genai.Text2ImagePipeline(model_dir, device="HETERO:GPU,CPU", **self.config)
            
            load_time = time.time() - start_time
            output_queue.put({'status': 'load_success', 'load_time': load_time})
            logging.info(f"Worker: T2I model loaded successfully in {load_time:.2f}s")
        except Exception as e:
            self.pipe = None
            error_info = f"T2I模型加载失败: {str(e)}\n{traceback.format_exc()}"
            output_queue.put({'status': 'error', 'message': error_info})
            logging.error(error_info)

    def unload_model(self, output_queue):
        """卸载T2I模型"""
        self.pipe = None
        self.model_name = None
        self.quant = None
        output_queue.put({'status': 'unload_success'})
        logging.info("Worker: T2I model unloaded.")

    def generate(self, params, output_queue):
        """生成图像"""
        if not self.pipe:
            output_queue.put({'status': 'error', 'message': 'T2I model not loaded.'})
            return

        output_dir = os.path.join(os.getcwd(), 'pictures')
        os.makedirs(output_dir, exist_ok=True)

        num_images = params.get('num_images', 1)
        initial_seed = params.get('seed')
        num_inference_steps = params['num_inference_steps']

        for i in range(num_images):
            try:
                output_queue.put({'status': 'progress', 'data': f"\n--- 开始生成第 {i + 1}/{num_images} 张图像 ---\n"})
                
                current_seed = initial_seed + i if initial_seed is not None else random.randint(1, 100000)
                generator = ov_genai.TorchGenerator(current_seed)

                # 创建tqdm进度条
                pbar = tqdm.tqdm(total=num_inference_steps, desc=f"生成图像 {i + 1}/{num_images}", 
                                file=sys.stdout, leave=False)

                def callback(step, num_steps, latent):
                    # 更新tqdm进度条
                    pbar.update(1)
                    # The callback gives us the step number (0-indexed) and total steps.
                    # We send this to the UI to construct the progress bar there.
                    output_queue.put({'status': 't2i_progress_update', 'step': step, 'total': num_steps, 'image_num': i + 1, 'total_images': num_images})
                    return False

                start_time = time.time()
                image_tensor = self.pipe.generate(
                    prompt=params['prompt'],
                    negative_prompt=params.get('negative_prompt', ""),
                    width=params.get('width', 512),
                    height=params.get('height', 512),
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    callback=callback
                )
                generation_time = time.time() - start_time

                # 关闭进度条
                pbar.close()

                # 保存图像
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"t2i_{timestamp}_{i + 1}.png"
                filepath = os.path.join(output_dir, filename)

                import numpy as np
                img_data = image_tensor.data[0]
                if img_data.ndim == 4: img_data = np.squeeze(img_data, axis=0)
                if img_data.max() <= 1.0: img_data = (img_data * 255).astype(np.uint8)
                Image.fromarray(img_data).save(filepath)

                output_queue.put({'status': 'image_generated', 'path': filepath})
                output_queue.put({'status': 'progress', 'data': f"图像已保存: {filepath}\n"})
                logging.info(f"Worker: Image saved to {filepath}")

            except Exception as e:
                # 确保进度条关闭
                if 'pbar' in locals():
                    pbar.close()
                error_info = f"T2I生成失败 (图片 {i + 1}/{num_images}): {str(e)}\n{traceback.format_exc()}"
                output_queue.put({'status': 'error', 'message': error_info})
                logging.error(error_info)
                continue
        
        output_queue.put({'status': 'generate_success'})


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


def t2i_worker_process(input_queue, output_queue):
    """T2I工作进程的主函数"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    worker = T2IWorker()
    logging.info("T2I Worker process started.")

    while True:
        try:
            task = input_queue.get()
            if task is None:
                logging.info("T2I Worker process shutting down.")
                break

            command = task.get('command')
            if command == 'load':
                worker.load_model(task['model_name'], task['quant'], output_queue)
            elif command == 'unload':
                worker.unload_model(output_queue)
            elif command == 'generate':
                worker.generate(task['params'], output_queue)
            else:
                logging.warning(f"Unknown command: {command}")
        except Exception as e:
            error_info = f"T2I Worker process error: {str(e)}\n{traceback.format_exc()}"
            output_queue.put({'status': 'error', 'message': error_info})
            logging.error(error_info)
