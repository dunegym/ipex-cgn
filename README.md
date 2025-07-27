# ipex-cgn

## 项目功能

基于openvino_genai的本地生成式模型应用平台，提供图形化界面，可以调用CPU、核显（NPU模块正在开发中）。

## 硬件要求

本项目已通过验证的硬件环境为Intel Ultra 5 125H处理器；理论上，酷睿i系列12代至14代处理器，及酷睿Ultra全系列处理器均可运行本项目，欢迎大家用更多元的硬件来体验并反馈问题。

## 软件要求

作者推荐使用最新版Anaconda，并用conda管理环境与第三方库，缺环境的用户可以前往 https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/ 下载安装；venv也是可以使用的，如Intel官方文档所示： https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html

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
若使用venv，则
```bash
python -m venv openvino-cgn
```

#### 第四步：激活环境
```bash
conda activate openvino-cgn
```
若使用venv，则
```bash
openvino-cgn/Scripts/activate
```

#### 第五步：安装依赖
```bash
pip install -r requirements.txt
```

#### 第六步：运行主程序
```bash
python main.py
```

第一次环境配置完成后，运行run.bat即可再次启动，无需切换目录、激活环境等步骤。（run.bat针对conda编写，使用venv需要修改脚本）

## 项目说明

受GitHub文件大小限制，模型文件没有上传至仓库，大家可以通过UI中的下载界面跳转至下载地址。有技术力与探索欲的用户可以自行量化更多模型。

从迅雷网盘下载好压缩包后，解压得到模型文件夹，结构应类似：
```
Qwen2.5-3B/
├─int4/
└─int8/
```
或
```
dreamlike-anime/
├─fp16/
└─int8/
```
在项目根目录（即ipex-cgn文件夹）下新建一个文件夹model，将模型文件夹移动到其下即可，最终项目文件夹结构应形如：
```
ipex-cgn
├─logs/
├─model/
│  ├─dreamlike-anime/
│  │  ├─fp16/
│  │  └─int8/
│  ├─Qwen2.5-3B/
│  │  ├─int4/
│  │  └─int8/
|  ......
├─__pycache__/
├─config.py
├─hetero_config.xml
├─LICENSE
......
└─ui.py
```

在Intel官方的努力下，openvino-genai库及其依赖正处于不断更新中，本人也将及时跟进，使该项目能支持当下最流行的大模型。😀

## 拓展教程

### 导出、量化模型

#### 即下载即导出

对于大语言模型，输入命令：

```bash
optimum-cli export openvino -m 仓库持有者/模型名 --weight-format int4（可替换为int8） --sym --ratio 1.0 --group-size 128 导出目录名
```

例如：

```bash
optimum-cli export openvino -m deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --weight-format int4 --sym --ratio 1.0 --group-size 128 DeepSeek-R1-1.5B-int4
```

对于文生图模型，输入命令：

```bash
optimum-cli export openvino --model 仓库持有者/模型名 --weight-format int8（可替换为fp16） 导出目录名
```

例如：

```
optimum-cli export openvino --model dreamlike-art/dreamlike-anime-1.0 --weight-format fp16 dreamlike-anime-fp16
```

#### 从本地仓库导出

对于大语言模型，输入命令：

```bash
optimum-cli export openvino -m 本地仓库路径 --weight-format int4（可替换为int8） --sym --ratio 1.0 --group-size 128 --task text-generation-with-past 导出目录名
```

例如：

```bash
optimum-cli export openvino -m E:\Model\LLM\HuggingFace\TinyLlama-1.1B-Chat-v1.0 --weight-format int4 --sym --ratio 1.0 --group-size 128 --task text-generation-with-past TinyLlama-1.1B-int4
```

对于文生图模型，输入命令：

```
optimum-cli export openvino --model 本地仓库路径 --weight-format fp16（可替换为int8） --task text-to-image 导出目录名

```

例如：

```
optimum-cli export openvino --model E:/Model/picture/dreamlike-photoreal-2.0 --weight-format fp16 --task text-to-image dreamlike-anime-fp16

```

### 使用新模型

第一步：将新模型文件夹移动至"ipex-cgn/model/模型名"文件夹下。

第二步：重命名为量化精度，如int4、int8、fp16。

第三步：修改config.py，将新模型名添加至模型字典（LLM、生图模型等不同功能的模型对应的字典不同，须注意区分），键为模型名，若无分享需求则值可为任意字符串。

第四步：保存后即生效。

## 引用

快速入门：https://github.com/openvinotoolkit/openvino.genai

openvino_genai库官方文档：https://docs.openvino.ai/2025/api/genai_api/_autosummary/openvino_genai.html

DeepSeek-R1-Distill-Qwen-1.5B模型文档：https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

Qwen2.5-3B-Instruct模型文档：https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

DeepSeek-R1-Distill-Qwen-7B模型文档：https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

Qwen2.5-Coder-3B-Instruct模型文档：https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct

dreamlike-anime-1.0模型文档：https://huggingface.co/dreamlike-art/dreamlike-anime-1.0

dreamlike-photoreal-2.0模型文档：https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0

dreamlike-diffusion-1.0模型文档：https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0