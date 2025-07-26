import logging

# LLM模型字典，包含模型名称和下载链接
LLM_MODEL_DICT = {
    'DeepSeek-R1-1.5B': 'https://pan.xunlei.com/s/VOVhbCCHgyEyu-dra2UjmHQ5A1?pwd=qkc7#',
    'DeepSeek-R1-7B': 'https://pan.xunlei.com/s/VOVhbnljD22zWV5oqAn9bYEdA1?pwd=hy84#',
    'Qwen2.5-3B': 'https://pan.xunlei.com/s/VOVhc2gr-9Ph8l8wQPRc4kcrA1?pwd=z9sn#', 
    'Qwen2.5-Coder-3B': 'https://pan.xunlei.com/s/VOVhcCYnrCI1e6xO09VLD5-2A1?pwd=9265#'
}

# LLM模型支持的量化精度列表
LLM_QUANTIZATION_LIST = [
    'int8', 
    'int4'
]

# LLM模型支持的设备列表
LLM_DEVICE_LIST = [
    'CPU',
    'GPU'
]

# 文生图模型字典，包含模型名称和下载链接
T2I_MODEL_DICT = {
    'dreamlike-anime': 'https://pan.xunlei.com/s/VOVh__wSi9g4VMDQ4xA575RkA1?pwd=3rif#',
    'dreamlike-photoreal': 'https://pan.xunlei.com/s/VOVh_ozxL8qAx5s7vkZGk3i0A1?pwd=6qcw#',
    'dreamlike-diffusion': 'https://pan.xunlei.com/s/VOVh_xz5uv86aEgbYNzS8Yr-A1?pwd=b5bf#'
}

# 文生图模型支持的量化精度列表
T2I_QUANTIZATION_LIST = [
    'fp16',
    'int8'
]

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
