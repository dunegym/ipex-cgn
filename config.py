import logging

MODEL_LIST = ['DeepSeek-R1-1.5B']
QUANTIZATION_LIST = ['int8', 'nf4', 'int4']
DEVICE_LIST = ['CPU', 'GPU']

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
