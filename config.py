import logging

LLM_MODEL_LIST = [
    'DeepSeek-R1-1.5B',
    'DeepSeek-R1-7B',
    'Qwen2.5-3B', 
    'Qwen2.5-Coder-3B'
]
LLM_QUANTIZATION_LIST = [
    'int8', 
    'int4'
]
LLM_DEVICE_LIST = [
    'CPU', 
    'GPU'
]
T2I_MODEL_LIST = [
    'dreamlike-anime',
    'dreamlike-photoreal',
    'dreamlike-diffusion'
]
T2I_QUANTIZATION_LIST = [
    'fp16',
    'int8'
]



logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
