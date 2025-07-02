import logging

MODEL_LIST = ['DeepSeek-R1']
QUANTIZATION_LIST = ['int4', 'int8']
DEVICE_LIST = ['CPU', 'GPU']

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
