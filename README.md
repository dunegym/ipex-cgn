# ipex-cgn

## 项目功能

基于openvino_genai的本地LLM聊天平台，提供图形化界面，可以在CPU、核显、集成NPU之间自如切换。

## 硬件要求

本项目已通过验证的硬件环境为Intel Ultra 5 125H处理器；理论上搭载Intel NPU的Meteor Lake、Arrow Lake及Lunar Lake系列处理器均可运行本项目，欢迎大家用更多元的硬件来体验并反馈问题。

## 软件要求

本人推荐使用最新版Anaconda，并用conda管理环境与第三方库，缺环境的读者可以前往 https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/ 下载安装；venv也是可以使用的，如Intel官方文档所示： https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html

## 部署方法

#### 第一步：克隆仓库
```bash
git clone https://github.com/dunegym/ipex-cgn.git
```

#### 第二步：切换到项目目录
```bash
cd ipex-cgn
```

#### 第三步：创建 Conda 环境
```bash
conda create -n openvino-cgn python=3.12
```

#### 第四步：激活环境
```bash
conda activate openvino-cgn
```

#### 第五步：安装依赖
```bash
pip install nncf==2.14.1 onnx==1.17.0 optimum-intel==1.22.0
pip install openvino==2025.1 openvino-tokenizers==2025.1
openvino-genai==2025.1
pip install PyQt5
```

#### 第六步：运行主程序
```bash
python main.py
```

##### 第一次环境配置完成后，运行run.bat即可再次启动，无需切换目录、激活环境等步骤。

## 项目说明

受GitHub文件大小限制，只有TinyLlama-1.1B-Chat-v1.0与DeepSeek-R1-Distill-Qwen-1.5B两款模型被上传至仓库；大家可以自行量化更多LLM。

在Intel官方的努力下，openvino-genai库及其依赖正处于不断更新中，本人也将及时跟进，使该项目能支持当下最流行的LLM。😀

##### 模型实测

TinyLlama-1.1B-Chat-v1.0：可以在任意芯片上正常使用。

DeepSeek-R1-Distill-Qwen-1.5B：可以在CPU与GPU上正常使用；int4量化模型能在NPU上使用，但使用缓存时会乱码；int8量化模型无法在NPU上使用。

ChatGLM3-6B：可以在CPU与GPU上正常使用，在NPU上会乱码。

Qwen3-4B：chat template异常，无法使用。

## 拓展教程

### 导出、量化模型

#### 即下载即导出

输入命令：

```bash
optimum-cli export openvino -m 仓库持有者/模型名 --weight-format int4（可替换为int8） --sym --ratio 1.0 --group-size 128 导出目录名
```

例如：

```bash
optimum-cli export openvino -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int4 --sym --ratio 1.0 --group-size 128 TinyLlama-1.1B-Chat-v1.0
```

#### 从本地仓库导出

输入命令：

```bash
optimum-cli export openvino -m 本地仓库路径 --weight-format int4（可替换为int8） --sym --ratio 1.0 --group-size 128 --task text-generation-with-past 导出目录名
```

例如：

```bash
optimum-cli export openvino -m E:\Model\LLM\HuggingFace\TinyLlama-1.1B-Chat-v1.0 --weight-format int4 --sym --ratio 1.0 --group-size 128 --task text-generation-with-past TinyLlama-1.1B-int4
```

### 使用新模型

###### 第一步：将新模型文件夹移动至ipex-cgn/model文件夹下。

###### 第二步：重命名，命名规则为“模型名-量化精度”，例如“TinyLlama-1.1B-int4”，其中TinyLlama-1.1B为模型名，int4为量化精度。

###### 第三步：修改main.py第9行，将新模型名添加至model_list末尾。

###### 第四步：保存后即生效。

### 体验更前沿的模型

输入命令：

```bash
pip install --upgrade optimum-intel
```

optimum-intel及其依赖（包括transformers等在内）将得到升级，近期流行的LLM均可支持。

## 引用

快速入门：https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html

openvino_genai库官方文档：https://docs.openvino.ai/2025/api/genai_api/_autosummary/openvino_genai.html

TinyLlama-1.1B模型文档：https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

DeepSeek-R1-Distill-Qwen-1.5B模型文档：https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
