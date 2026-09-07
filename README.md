# ovtool — 综合性 OpenVINO 命令行工具

基于 [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) 的多功能推理命令行工具，支持：

- **大语言模型（LLM）推理**：单轮生成 + 多轮交互聊天，流式输出，完整采样参数
- **多模态（VLM）推理**：图像 + 文本问答（LLaVA / Qwen-VL / MiniCPM-V / InternVL 等转换后模型）
- **图像生成**：文生图（Text2Image）与图生图（Image2Image），支持 SD / SDXL / Flux 系列
- **设备选择与运行时参数**：CPU / GPU / NPU / AUTO / HETERO，透传 OpenVINO 运行时选项
- **模型转换与量化**：Hugging Face 模型一键导出 OpenVINO IR，支持 INT8 / INT4 权重压缩（含 AWQ）

## 环境

```bash
conda create -n openvino-cli python=3.11 -y -c conda-forge --override-channels
conda activate openvino-cli
pip install -e .            # 推理所需（openvino / openvino-genai / openvino-tokenizers）
pip install "optimum-intel[openvino]" onnx   # 转换/量化所需（或 pip install -e ".[convert]"）
```

安装后命令入口为 `ovtool`（等价于 `python -m ovtool.cli`）。

## 快速上手

```bash
# 1. 查看可用推理设备
ovtool devices

# 2. 转换 + INT4 量化一个 LLM（下载自 Hugging Face）
ovtool convert llm Qwen/Qwen2.5-0.5B-Instruct -m ./qwen05-int4

# 3. 单轮生成
ovtool generate -m ./qwen05-int4 -d CPU "用一句话介绍 OpenVINO"

# 4. 多轮交互聊天
ovtool chat -m ./qwen05-int4 -d GPU --opt perf_mode=LOW_LATENCY
```

## 子命令详解

### `ovtool devices`

列出 OpenVINO 可用设备及完整设备名、驱动版本、子设备（如 `GPU.0 / GPU.1`）。

### `ovtool convert <kind> <model>`

| kind | 模型家族 | 默认导出任务 |
|---|---|---|
| `llm` | 文本大模型 | `text-generation-with-past` |
| `vlm` | 视觉语言模型 | `image-text-to-text` |
| `image` | 扩散图像生成 | 自动按模型选择（SD / SDXL / Flux / LCM） |

常用参数：

- `-o DIR` 输出目录（默认 `./<模型名>-<量化格式>`）
- `--weight-format`：`fp32` / `fp16` / `int8` / `int4` / `int4_symg128`（对称分组 128）等预设
- `--sym` / `--asym`：对称 / 非对称量化（**跑 NPU 建议对称 int4**；CPU/GPU 用非对称精度更好）
- `--ratio 0.8`、`--group-size 64`：int4 压缩比例与分组大小
- `--awq --dataset wikitext2`：激活感知量化（AWQ）
- `--trust-remote-code`：允许执行 HF 仓库自定义建模代码

示例：

```bash
ovtool convert vlm openbmb/MiniCPM-V-2_6 -m ./minicpmv-int4 --sym
ovtool convert image stabilityai/sd-turbo -m ./sd-turbo-ir --weight-format int8
```

### `ovtool generate` / `ovtool chat`（LLM）

公共参数：`-m` 模型目录；`-d` 设备（`CPU`/`GPU`/`NPU`/`AUTO`/`HETERO:GPU,CPU`…）；
`--opt KEY=VALUE` 运行时选项（可重复），如：

- `--opt perf_mode=THROUGHPUT|LOW_LATENCY|CUMULATIVE_THROUGHPUT`
- `--opt inference_num_threads=8`
- `--opt num_streams=auto`

生成参数：`--max-new-tokens` `--temperature`（>0 启用采样）`--top-p` `--top-k`
`--repetition-penalty` `--rng-seed` `--stop-tokens`；`--no-stream` 关闭流式输出；`--stats` 打印 TTFT/TPOT/吞吐。

**NPU 专属参数**（`-d NPU` 时自动生效默认值）：`--max-prompt-len`（默认 1024）与
`--min-response-len`（默认 128）设定静态形状编译预算；NPU 模型必须用
`ovtool convert llm ... --weight-format int4 --sym` 转换（对称量化）。

聊天模式内置命令：`/exit` 退出，`/reset` 清空历史，`/system <text>` 设置系统提示词。

### `ovtool vlm`（多模态）

```bash
ovtool vlm -m ./minicpmv-int4 -d GPU -i ./photo.jpg "描述这张图片的内容"
```

`-i` 可重复传多张图；生成参数与 LLM 相同。

### `ovtool image` / `ovtool image2image`（扩散模型）

```bash
ovtool image -m ./sd-turbo-ir -d GPU "a corgi surfing a wave" \
    --width 512 --height 512 --steps 8 --guidance-scale 1.0 --seed 42 \
    --out-dir ./generated
```

参数：`--width/--height`、`--steps`（去噪步数）、`--guidance-scale`、`--num-images`、
`--negative-prompt`、`--seed`、`--scheduler`（如 `LCM`、`EULER_ANCESTRAL`，需模型适配）、`--out-dir`。

> 注意：扩散类模型官方推荐在 **GPU** 上运行（默认设备即 GPU）。
> NPU 上扩散模型为分段执行（text encoder + UNet 在 NPU、VAE decoder 放 GPU），本工具暂不自动编排该模式。

## 设备选择参考

| 设备 | 适用场景 | 注意事项 |
|---|---|---|
| CPU | 通用，AVX2/AVX-512/AMX 加速 | LLM INT4/INT8 均可 |
| GPU（iGPU / Arc / DC GPU） | 扩散模型最佳；LLM 吞吐好 | 需要 Intel 显卡驱动 |
| NPU（Core Ultra） | LLM 低功耗推理 | **必须对称 INT4（`convert ... --sym`）**；静态形状执行，用 `--max-prompt-len`（默认 1024）/`--min-response-len`（默认 128）设定编译期形状；本机 NPU 3720 实测 Qwen3-0.6B 约 21 tok/s |
| AUTO / HETERO | 自动选择 / 混合执行 | 适合不确定设备能力时 |

## 代码结构

```
ovtool/
├── cli.py        # 入口与子命令注册
├── devices.py    # 设备枚举/校验
├── convert.py    # optimum-intel 导出 + 权重量化
├── llm.py        # LLMPipeline：generate / chat
├── vlm.py        # VLMPipeline：图文多模态
└── imagegen.py   # Text2Image / Image2Image
```

## 已验证（本机：Core Ultra 5 125H + Arc Pro iGPU + NPU 3720）

- `devices` / 全部子命令 `--help`：✅
- LLM 转换 + INT4 量化（Qwen2.5-0.5B-Instruct，322MB int4 IR）：✅
- LLM 转换 + INT4 量化（**Qwen3-0.6B**，思考模型）：✅ GPU 生成 ~55 tok/s，CPU 多轮聊天正常，数学比较回答正确
- `generate` 单轮生成（CPU / GPU，流式与非流式，`--stats`、`--opt perf_mode=...`）：✅（iGPU 上 ~50 tok/s @0.5B-int4）
- `chat` 多轮交互（含 `/exit` `/reset` `/system`）：✅
- 扩散模型转换 + INT8 量化（sd-turbo）：✅
- `image` 文生图 / `image2image` 图生图（GPU，seed 复现）：✅（512×512×4 步约数秒）
- **NPU 推理（Qwen3-0.6B 对称 INT4）**：✅ TTFT ~1.5s，~21 tok/s；`--max-prompt-len` / `--min-response-len` 静态形状参数生效
- **Qwen3.5-0.8B / Qwen3.5-2B**（新一代原生多模态架构 `qwen3_5`，对称/非对称 INT4）：✅ 纯文本生成 GPU 正常（2B ~33 tok/s；0.8B NPU 编译极慢）
- **Qwen3-VL-2B-Instruct INT4**：✅ 纯文本生成正常
- `vlm` 多模态路径已按 openvino-genai 官方 API 实现，未做整机下载验证（需转换 VLM 模型后使用）

## 已知限制（2026-09 实测）

1. **Qwen3-VL / Qwen3.5 的图像输入**：转换成功但 `VLMPipeline` 图文推理在 GenAI 2026.3.1 上报 `Argument shapes are inconsistent`（各尺寸均复现）；master 分支已有专门的 `InputsEmbedderQwen3VL/Qwen3_5` 实现，等待新版本发布。**纯文本模式不受影响**。
2. **VLM 跑 NPU**：Qwen3.5-0.8B 文本模式在 NPU 上长时间编译后触发 `ZE_RESULT_ERROR_DEVICE_LOST`（驱动挂死，需重启进程恢复）。NPU 上建议只跑对称 INT4 的纯 LLM（qwen3-0.6b-sym 已验证）。
3. **FLUX.2-klein 导出**：optimum-intel 2.1.0 追踪 `pos_embed` 时报 `Axis out of rank range`（上游 [issue #1767](https://github.com/huggingface/optimum-intel/issues/1767) 跟踪中）。图像生成请用 SD/SDXL/Flux.1 系（sd-turbo 已验证）。
4. **optimum 版本护栏**：optimum-intel 2.1.0 对 qwen3-vl/qwen2-vl 等新架构钉了过时的 `MAX_TRANSFORMERS_VERSION`，本工具 `convert vlm` 已自动放宽（`_relax_stale_version_guards`）；qwen3_5 还要求 transformers==5.2.x（5.3+ 移除了 `Qwen3_5DynamicCache`，而 optimum 钉 `<5.6`，5.2 恰好同时满足）。

## 实现备注（踩坑记录）

- **扩散模型的分词器位置**：GenAI 的 SD 管线按 `text_encoder → tokenizer` 路径约定，在 `tokenizer/` 子目录查找 `openvino_tokenizer.xml`；`ovtool convert image` 已自动放置到位（LLM/VLM 仍在根目录）。
- **openvino_tokenizers 扩展**：推理模块统一预 `import openvino_tokenizers` 注册自定义算子，避免 Tokenizer 加载失败。
- **CLIP 类慢分词器**转换需要 `sentencepiece` / `tiktoken`（已加入依赖）。
- 生成参数 `rng_seed`、图像结果返回 `ov.Tensor(N,H,W,C)` 均按 openvino-genai 2026.3 API 适配。
- optimum-intel 2.1.0 不随 `save_pretrained` 保存转换后分词器，故 `convert` 内置了 openvino-tokenizers 转换与保存。
- **tiktoken 后端分词器的 `tokenizer.json` 序列化损坏**：transformers 5.x 下 Qwen3 等新模型的分词器 `save_pretrained` 写出的 `tokenizer.json` 用 tokenizers 库编码得到空结果（静默坏文件）。`_save_tokenizer` 现在优先从原始 HF 仓库加载分词器，并对每次加载做非空编码探针校验。
