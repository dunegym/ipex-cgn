import logging

LLM_MODEL_LIST = ['DeepSeek-R1-1.5B']
LLM_DOWNLOAD_LIST = ['Qwen2.5-3B', 
                     'Qwen2.5-Coder-3B', 
                     'DeepSeek-R1-7B']
QUANTIZATION_LIST = ['int8', 
                     'int4']
DEVICE_LIST = ['CPU', 'GPU']


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
