import os
import time
import logging
from transformers import AutoTokenizer
import openvino_genai as ov_genai
from config import LLM_QUANTIZATION_LIST

class LLMChatManager:
    """管理LLM聊天的类"""
    def __init__(self):
        # 初始化管道和分词器
        self.pipe = None
        self.tokenizer = None
        self.chat_history = []  # 聊天历史记录

    def clear_history(self):
        """清空聊天历史记录"""
        self.chat_history = []

    def load_model(self, model_name, quant, device, console_callback=None):
        """加载指定的模型"""
        model_dir = f"model/{model_name}/{quant}"  # 更新路径结构
        logging.info(f"开始加载模型: {model_name}, 量化精度: {quant}, 设备: {device}")
        if console_callback:
            console_callback('Loading......\n')
            # 添加短暂延时确保信号处理
            time.sleep(0.1)
        
        try:
            start_time = time.time()
            if console_callback:
                console_callback(f"正在准备模型目录: {model_dir}...\n")
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
                    console_callback(f"正在加载NPU模型，这可能需要一些时间...\n")
                    time.sleep(0.1)
                    
                self.pipe = ov_genai.LLMPipeline(
                    model_dir,
                    device,
                    CACHE_DIR=f".npucache/{model_name}/{quant}",  # 更新缓存路径结构
                    MAX_PROMPT_LEN=4096
                )
            else:
                # 如果设备不是NPU，直接加载模型
                if console_callback:
                    console_callback(f"正在加载模型到{device}设备，这可能需要一些时间...\n")
                    time.sleep(0.1)
                    
                self.pipe = ov_genai.LLMPipeline(model_dir, device)

            # 加载分词器
            if console_callback:
                console_callback("模型加载完成，正在加载分词器...\n")
                time.sleep(0.1)
                
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=True
            )
            end_time = time.time()
            load_time = end_time - start_time
            if console_callback:
                console_callback(f"模型加载成功！耗时：{load_time:.2f} 秒\n\n")
            logging.info(f"模型加载成功: {model_name}, 耗时: {load_time:.2f} 秒")
            return True
        except Exception as e:
            # 如果加载失败，清空管道和分词器
            self.pipe = None
            self.tokenizer = None
            self.handle_exception(e, console_callback, prefix="模型加载失败")
            return False

    def unload_model(self, console_callback=None):
        """卸载模型"""
        logging.info("开始卸载模型")
        try:
            self.pipe = None
            self.tokenizer = None
            if console_callback:
                console_callback("模型已成功卸载！\n\n")
            logging.info("模型卸载成功")
            return True
        except Exception as e:
            self.handle_exception(e, console_callback, prefix="模型卸载失败")
            return False

    def build_prompt(self, user_input, model_name):
        """构建聊天提示"""
        model_dir = None
        for quant in LLM_QUANTIZATION_LIST:
            candidate = f"model/{model_name}/{quant}"  # 更新路径结构
            if os.path.isdir(candidate):
                model_dir = candidate
                break
        if model_dir is None:
            model_dir = f"model/{model_name}/int4"  # 默认使用int4量化模型

        try:
            n = len(self.chat_history)
            history = self.chat_history[-n:] if n <= 6 else self.chat_history[-6:]  # 限制历史记录长度为6
            messages = [
                {"role": "system", "content": "You are a helpful assistant. "}
            ] + history + [{"role": "user", "content": user_input}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            self.handle_exception(e, prefix="Prompt 构建失败")
            return f"<|user|>\n{user_input}\n<|assistant|>\n"

    def generate_reply(self, prompt, window, max_new_tokens=2048):
        """生成回复"""
        if not self.pipe:
            err = RuntimeError("模型未加载")
            self.handle_exception(err)
            raise err
        try:
            device = getattr(self.pipe, 'device', None)
            do_sample = False if device and str(device).upper() == 'NPU' else True  # 根据设备类型设置采样方式
            streamer = lambda x: window.chat_callback(x)  # 定义流式输出回调
            result = self.pipe.generate(
                [prompt],
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True
            )
            perf_metrics = result.perf_metrics
            throughput = perf_metrics.get_throughput().mean if hasattr(perf_metrics, 'get_throughput') else '无法统计'
            logging.info(f"生成回复完成, 速度: %.2f tokens/s"%(throughput))
            return result
        except Exception as e:
            self.handle_exception(e, prefix="推理失败")
            raise

    def append_history(self, user_input, assistant_output):
        """将用户输入和助手输出添加到聊天历史记录中"""
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": str(assistant_output)})
        if len(self.chat_history) > 6:
            self.chat_history = self.chat_history[-6:]  # 保持历史记录长度不超过6

    def handle_exception(self, e, console_callback=None, prefix="错误"):
        """处理异常并记录日志"""
        msg = f"{prefix}: {str(e)}\n\n"
        logging.error(msg)
        if console_callback:
            console_callback(msg)