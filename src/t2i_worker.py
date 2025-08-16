import time
import logging
import openvino_genai as ov_genai
import traceback
import os
import random
import tqdm
import sys
from PIL import Image

# --- Absolute Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

class T2IWorker:
    """处理T2I相关任务的工作类"""
    def __init__(self):
        self.pipe = None
        self.model_name = None
        self.quant = None
        self.config = {
            "HETERO_CONFIG_FILE": os.path.join(project_root, "config", "hetero_config.xml"),
            "PERFORMANCE_HINT": "THROUGHPUT"
        }

    def load_model(self, model_name, quant, output_queue):
        """加载T2I模型"""
        self.model_name = model_name
        self.quant = quant
        model_dir = os.path.join(project_root, "model", model_name, quant)
        logging.info(f"Worker: Loading T2I model {model_dir}")
        try:
            output_queue.put({'status': 'progress', 'data': 'Loading T2I model......\n'})
            start_time = time.time()
            
            output_queue.put({'status': 'progress', 'data': f"正在准备T2I模型目录: {model_dir}...\n"})
            output_queue.put({'status': 'progress', 'data': "开始加载T2I模型...\n"})
            
            self.pipe = ov_genai.Text2ImagePipeline(model_dir, device="HETERO:GPU,CPU", **self.config)
            output_queue.put({'status': 'progress', 'data': "T2I模型加载完成。\n"})
            
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

        output_dir = os.path.join(project_root, 'pictures')
        os.makedirs(output_dir, exist_ok=True)

        num_images = params.get('num_images', 1)
        initial_seed = params.get('seed')
        num_inference_steps = params['num_inference_steps']
        width = min(params.get('width', 512), 512)  # Limit width to 512
        height = min(params.get('height', 512), 512)  # Limit height to 512

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
                pbar.close()
                output_queue.put({'status': 'progress', 'data': f"第 {i + 1}/{num_images} 张图像生成耗时: {generation_time:.2f} 秒.\n"})

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
