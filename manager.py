import os
import time
import logging
from transformers import AutoTokenizer
import openvino_genai as ov_genai
import random
from config import LLM_QUANTIZATION_LIST
from PIL import Image

class LLMChatManager:
    """管理LLM聊天的类"""
    def __init__(self):
        # 初始化管道和分词器
        self.llm_pipe = None
        self.llm_tokenizer = None
        self.llm_chat_history = []  # 聊天历史记录

    def llm_clear_history(self):
        """清空聊天历史记录"""
        self.llm_chat_history = []

    def llm_load_model(self, model_name, quant, device, console_callback=None):
        """加载指定的模型"""
        model_dir = f"model/{model_name}/{quant}"  # 更新路径结构
        logging.info(f"开始加载LLM模型: {model_name}, 量化精度: {quant}, 设备: {device}")
        if console_callback:
            console_callback('Loading LLM model......\n')
            # 添加短暂延时确保信号处理
            time.sleep(0.1)
        
        try:
            start_time = time.time()
            if console_callback:
                console_callback(f"正在准备LLM模型目录: {model_dir}...\n")
                time.sleep(0.1)
                
            if device == 'NPU':
                # 如果设备是NPU，创建缓存目录
                if not os.path.exists('.npucache'):
                    os.makedirs('.npucache')
                if console_callback:
                    console_callback(f"为NPU设备创建缓存目录...\n")
                    time.sleep(0.1)
                
                # 加载前通知
                if console_callback:
                    console_callback(f"正在加载NPU LLM模型，这可能需要一些时间...\n")
                    time.sleep(0.1)
                    
                self.llm_pipe = ov_genai.LLMPipeline(
                    model_dir,
                    device,
                    CACHE_DIR=f".npucache/{model_name}/{quant}",  # 更新缓存路径结构
                    MAX_PROMPT_LEN=4096
                )
            else:
                # 如果设备不是NPU，直接加载模型
                if console_callback:
                    console_callback(f"正在加载LLM模型到{device}设备，这可能需要一些时间...\n")
                    time.sleep(0.1)
                    
                self.llm_pipe = ov_genai.LLMPipeline(model_dir, device)

            # 加载分词器
            if console_callback:
                console_callback("LLM模型加载完成，正在加载分词器...\n")
                time.sleep(0.1)
                
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=True
            )
            end_time = time.time()
            load_time = end_time - start_time
            if console_callback:
                console_callback(f"LLM模型加载成功！耗时：{load_time:.2f} 秒\n\n")
            logging.info(f"LLM模型加载成功: {model_name}, 耗时: {load_time:.2f} 秒")
            return True
        except Exception as e:
            # 如果加载失败，清空管道和分词器
            self.llm_pipe = None
            self.llm_tokenizer = None
            self.llm_handle_exception(e, console_callback, prefix="LLM模型加载失败")
            return False

    def llm_unload_model(self, console_callback=None):
        """卸载模型"""
        logging.info("开始卸载LLM模型")
        try:
            self.llm_pipe = None
            self.llm_tokenizer = None
            if console_callback:
                console_callback("LLM模型已成功卸载！\n\n")
            logging.info("LLM模型卸载成功")
            return True
        except Exception as e:
            self.llm_handle_exception(e, console_callback, prefix="LLM模型卸载失败")
            return False

    def llm_build_prompt(self, user_input, model_name):
        """构建聊天提示"""
        model_dir = None
        for quant in LLM_QUANTIZATION_LIST:
            candidate = f"model/{model_name}/{quant}"  # 更新路径结构
            if os.path.isdir(candidate):
                model_dir = candidate
                break
        if model_dir is None:
            model_dir = f"model/{model_name}/int4"  # 保底使用int4量化模型

        try:
            n = len(self.llm_chat_history)
            history = self.llm_chat_history[-n:] if n <= 6 else self.llm_chat_history[-6:]  # 限制历史记录长度为6
            messages = [
                {"role": "system", "content": "You are a helpful assistant. "}
            ] + history + [{"role": "user", "content": user_input}]
            return self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            self.llm_handle_exception(e, prefix="LLM Prompt 构建失败")
            return f"<|user|>\n{user_input}\n<|assistant|>\n"

    def llm_generate_reply(self, prompt, window, max_new_tokens=2048):
        """生成回复"""
        if not self.llm_pipe:
            err = RuntimeError("LLM模型未加载")
            self.llm_handle_exception(err)
            raise err
        try:
            device = getattr(self.llm_pipe, 'device', None)
            do_sample = False if device and str(device).upper() == 'NPU' else True  # 根据设备类型设置采样方式
            streamer = lambda x: window.chat_callback(x)  # 定义流式输出回调
            result = self.llm_pipe.generate(
                [prompt],
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True
            )
            perf_metrics = result.perf_metrics
            throughput = perf_metrics.get_throughput().mean if hasattr(perf_metrics, 'get_throughput') else '无法统计'
            logging.info(f"LLM生成回复完成, 速度: %.2f tokens/s"%(throughput))
            return result
        except Exception as e:
            self.llm_handle_exception(e, prefix="LLM推理失败")
            raise

    def llm_append_history(self, user_input, assistant_output):
        """将用户输入和助手输出添加到聊天历史记录中"""
        self.llm_chat_history.append({"role": "user", "content": user_input})
        self.llm_chat_history.append({"role": "assistant", "content": str(assistant_output)})
        if len(self.llm_chat_history) > 6:
            self.llm_chat_history = self.llm_chat_history[-6:]  # 保持历史记录长度不超过6

    def llm_handle_exception(self, e, console_callback=None, prefix="LLM错误"):
        """处理异常并记录日志"""
        msg = f"{prefix}: {str(e)}\n\n"
        logging.error(msg)
        if console_callback:
            console_callback(msg)

class T2IManager:
    """管理文生图功能的类"""
    def __init__(self):
        """初始化管道"""
        self.t2i_pipe = None
        self.t2i_model_name = None
        self.t2i_quant = None
        self.t2i_config = {
            "HETERO_CONFIG_FILE": os.path.join(os.getcwd(), "hetero_config.xml"),
            "PERFORMANCE_HINT": "THROUGHPUT"
        }

    def t2i_handle_exception(self, e, console_callback=None, prefix="T2I错误"):
        """处理异常并记录日志"""
        msg = f"{prefix}: {str(e)}\n\n"
        logging.error(msg)
        if console_callback:
            console_callback(msg)

    def t2i_load_model(self, model_name, quant, console_callback=None):
        """加载指定的文生图模型"""
        model_dir = f"model/{model_name}/{quant}"
        logging.info(f"开始加载T2I模型: {model_name}, 量化精度: {quant}")
        
        if console_callback:
            console_callback('Loading T2I model......\n')
            # 添加短暂延时确保信号处理
            time.sleep(0.1)
        
        try:
            start_time = time.time()
            if console_callback:
                console_callback(f"正在准备T2I模型目录: {model_dir}...\n")
                time.sleep(0.1)
                
            # 加载前通知
            if console_callback:
                console_callback(f"正在加载T2I模型到HETERO:GPU,CPU设备，这可能需要一些时间...\n")
                time.sleep(0.1)
                
            self.t2i_pipe = ov_genai.Text2ImagePipeline(
                model_dir,
                device="HETERO:GPU,CPU",
                **self.t2i_config
            )
            
            end_time = time.time()
            load_time = end_time - start_time
            if console_callback:
                console_callback(f"T2I模型加载成功！耗时：{load_time:.2f} 秒\n\n")
            logging.info(f"T2I模型加载成功: {model_name}, 耗时: {load_time:.2f} 秒")
            
            self.t2i_model_name = model_name
            self.t2i_quant = quant
            return True
            
        except Exception as e:
            # 如果加载失败，清空管道
            self.t2i_pipe = None
            self.t2i_model_name = None
            self.t2i_quant = None
            self.t2i_handle_exception(e, console_callback, prefix="T2I模型加载失败")
            return False

    def t2i_unload_model(self, console_callback=None):
        """卸载文生图模型"""
        logging.info("开始卸载T2I模型")
        try:
            self.t2i_pipe = None
            self.t2i_model_name = None
            self.t2i_quant = None
            if console_callback:
                console_callback("T2I模型已成功卸载！\n\n")
            logging.info("T2I模型卸载成功")
            return True
        except Exception as e:
            self.t2i_handle_exception(e, console_callback, prefix="T2I模型卸载失败")
            return False

    def t2i_generate_image(self, prompt, negative_prompt="", width=512, height=512, 
                          num_inference_steps=20, num_images=1, seed=None, 
                          console_callback=None, progress_callback=None):
        """生成图像"""
        if self.t2i_pipe is None:
            err = RuntimeError("T2I模型未加载")
            self.t2i_handle_exception(err)
            raise err
            
        try:
            # 设置随机种子 - 使用OpenVINO的随机生成器
            generator = None
            if seed is not None:
                generator = ov_genai.TorchGenerator(seed)
                if console_callback:
                    console_callback(f"使用种子: {seed}\n")
            else:
                seed = random.randint(1, 100000)
                generator = ov_genai.TorchGenerator(seed)
            
            if console_callback:
                console_callback(f"开始T2I生成图像...\n")
                console_callback(f"提示词: {prompt}\n")
                if negative_prompt:
                    console_callback(f"负向提示词: {negative_prompt}\n")
                console_callback(f"参数: {width}x{height}, {num_inference_steps}步\n")
            
            # 创建进度回调包装器
            def callback(step, num_steps, latent):
                if progress_callback:
                    progress_callback(step + 1, num_inference_steps)
                return False
            
            # 生成图像
            start_time = time.time()
            logging.info(f"开始T2I图像生成, 提示词: {prompt[:50]}...")
            
            try:
                image_tensor = self.t2i_pipe.generate(
                    prompt,
                    negative_prompt=negative_prompt if negative_prompt else "",
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images,
                    generator=generator,
                    callback=callback
                )
            except Exception as e1:
                if console_callback:
                    console_callback(f"尝试完整参数失败: {str(e1)}\n")
                logging.warning(f"T2I完整参数生成失败，使用简化参数: {str(e1)}")
                image_tensor = self.t2i_pipe.generate(
                    prompt,
                    negative_prompt=negative_prompt if negative_prompt else "",
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    callback=callback
                )
            
            generation_time = time.time() - start_time
            
            # 处理OpenVINO Tensor结果
            image_paths = []
            output_dir = os.path.join(os.path.dirname(__file__), 'pictures')
            os.makedirs(output_dir, exist_ok=True)
            
            if console_callback:
                console_callback(f"结果类型: {type(image_tensor)}\n")
            
            if hasattr(image_tensor, 'data'):
                images_data = image_tensor.data
                if console_callback:
                    console_callback(f"图像数据数量: {len(images_data)}\n")
                
                for i, img_data in enumerate(images_data):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"t2i_{timestamp}_{i+1}.png"
                    filepath = os.path.join(output_dir, filename)
                    
                    image = Image.fromarray(img_data)
                    image.save(filepath)
                    image_paths.append(filepath)
                    
                    if console_callback:
                        console_callback(f"图像已保存: {filepath}\n")
                    logging.info(f"T2I图像已保存: {filepath}")
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"t2i_{timestamp}_1.png"
                filepath = os.path.join(output_dir, filename)
                
                try:
                    if hasattr(image_tensor, 'numpy'):
                        img_array = image_tensor.numpy()
                    else:
                        import numpy as np
                        img_array = np.array(image_tensor)
                    
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    
                    image = Image.fromarray(img_array)
                    image.save(filepath)
                    image_paths.append(filepath)
                    
                    if console_callback:
                        console_callback(f"图像已保存: {filepath}\n")
                    logging.info(f"T2I图像已保存: {filepath}")
                        
                except Exception as convert_error:
                    if console_callback:
                        console_callback(f"图像转换失败: {str(convert_error)}\n")
                    logging.error(f"T2I图像转换失败: {str(convert_error)}")
                    raise convert_error
        
            if console_callback:
                console_callback(f"T2I生成完成！耗时: {generation_time:.2f}秒\n")
                console_callback(f"平均每步: {generation_time/num_inference_steps:.2f}秒\n\n")
            
            # 计算速度指标（类似LLM的throughput）
            steps_per_second = num_inference_steps / generation_time if generation_time > 0 else 0
            logging.info(f"T2I生成图像完成, 速度: {steps_per_second:.2f} steps/s")
            
            return image_paths
            
        except Exception as e:
            self.t2i_handle_exception(e, console_callback, prefix="T2I生成失败")
            raise

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)