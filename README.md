# ipex-cgn

## 项目功能

基于openvino-genai的本地LLM聊天平台，提供图形化界面，可以在CPU、核显、集成NPU之间自如切换。

## 硬件要求

本项目已通过验证的硬件环境为Intel Ultra 5 125H处理器；理论上搭载Intel NPU的Meteor Lake及Lunar Lake系列处理器均可运行本项目，欢迎大家用更多元的硬件来体验并反馈问题。

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
pip install openvino==2025.1 openvino-tokenizers==2025.1 openvino-genai==2025.1
```

#### 第六步：运行主程序
```bash
python main.py
```

## 项目说明

目前本项目只支持TinyLlama-1.1B模型，日后我会向仓库中添加更多量化好的的模型并更新主程序代码。为了方便使用，免去每次切换目录、激活环境的麻烦，我会将代码打包成exe文件并及时上传至仓库；exe版本将长期保持稳定，而main.py会持续更新。在Intel官方的努力下，openvino-genai库及其依赖正处于不断更新中，本人也将及时跟进，使该项目能支持当下最流行的LLM。😀