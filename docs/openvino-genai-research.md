# OpenVINO GenAI 软件栈研究报告：Intel 硬件设备 × 生成式模型支持矩阵

> 研究日期：2026-09-06
> 调研对象：[openvinotoolkit/openvino.genai](https://github.com/openvinotoolkit/openvino.genai)（及 OpenVINO Runtime、Optimum-Intel、OVMS、IPEX-LLM、Intel Gaudi 等相邻技术栈）
> 核心问题：**Intel 各类处理器（硬件设备）对各型生成式模型（模型类型/任务）的支持情况。**

---

## 摘要（TL;DR）

1. **OpenVINO GenAI 是构建在 OpenVINO Runtime 之上的生成式 AI 管线库**，提供 C++/Python/Node.js API；底层由 OpenVINO Runtime 的三个 device plugin（CPU / GPU / NPU）执行推理。官方设备口径始终是 **CPU、GPU、NPU 三类**，[GenAI README](https://github.com/openvinotoolkit/openvino.genai)、[OpenVINO Supported Devices 文档](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)。
2. **管线覆盖已非常全面**：LLM 文本生成、VLM 视觉语言模型、Whisper 语音识别、TTS（SpeechT5 / Kokoro）、文生图/图生图/重绘（SD 系 / SDXL / SD3.5 / Flux / FLUX.2 / Qwen-Image / Z-Image）、**视频生成（LTX-Video，Text2Video 与 Image2Video）**、Embeddings（含多模态 Qwen3-VL-Embedding）、Rerank，以及 vLLM 风格的 Continuous Batching（供 [OVMS](https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching.html) 服务化使用）。全部模型清单见官方 [Supported Models 页面](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)。
3. **设备 × 模型矩阵要点**：LLM/VLM/Whisper 三大核心管线官方明确支持 CPU + GPU + NPU 全平台；扩散类图像生成以 **GPU 为最佳**，NPU 上是"分段执行"（text encoder + 去噪在 NPU、VAE decoder 建议放 GPU）；TTS/视频生成官方文档未标注 NPU 支持。LLM 在 NPU 上有硬性量化要求（对称 INT4/NF4）和静态形状/prefill 分块的上下文限制。
4. **Intel Gaudi（Gaudi 2/3）不是 OpenVINO 的目标设备**：Gaudi 走 **SynapseAI（Habana 驱动/运行时）+ vLLM 硬件插件（vLLM Hardware Plugin for Intel Gaudi）+ Optimum-Habana** 路线，不存在官方 OpenVINO Gaudi device plugin（详见[下文](#与相邻技术栈的关系)）。
5. **IPEX-LLM 已归档**：[intel/ipex-llm 仓库](https://github.com/intel/ipex-llm)于 **2026-01-28 归档**（只读、不再接受补丁），Intel 客户端 LLM 路线事实上收敛到 OpenVINO/OVMS 栈。
6. **版本**：2025 年线的最新 release 为 **2025.4（2025.4.0.0，随 OpenVINO 2025.4 于 2025-12 发布）与 2025.4.1.0 补丁版**；截至本文调研日（2026-09-06），最新 release 已是 **2026.3.0.0（2026-08-05 发布）**（[Releases 页](https://github.com/openvinotoolkit/openvino.genai/releases)）。

---

## 1. 软件栈架构

### 1.1 分层结构

```
┌─────────────────────────────────────────────────────────────────┐
│  应用层: 用户 App / ComfyUI / LangChain / OpenAI 兼容客户端        │
├─────────────────────────────────────────────────────────────────┤
│  服务层: OVMS (OpenVINO Model Server)  ← 使用 GenAI Continuous    │
│          OpenAI 兼容 API (chat/completions, embeddings, rerank)   │  Batching + Paged Attention
├─────────────────────────────────────────────────────────────────┤
│  OpenVINO GenAI 库 (openvino-genai)                              │
│  LLMPipeline / VLMPipeline / WhisperPipeline / TTS /             │
│  Text2Image / Text2Video / EmbeddingPipeline / Rerank            │
│  + openvino-tokenizers (内置分词, 无外部依赖)                      │
│  + Continuous Batching 引擎、投机解码、LoRA、Sparse Attention      │
├─────────────────────────────────────────────────────────────────┤
│  OpenVINO Runtime (openvino, 独立仓库 openvinotoolkit/openvino)   │
├──────────────┬──────────────────────┬───────────────────────────┤
│  CPU plugin  │  GPU plugin (OpenCL) │  NPU plugin               │
│  (x86-64/ARM)│  iGPU / Arc / DC GPU │  Core Ultra NPU 3720      │
└──────────────┴──────────────────────┴───────────────────────────┘
```

- OpenVINO GenAI 官方定位："the most popular Generative AI model pipelines, optimized execution methods, and samples"，**所有场景都运行在 OpenVINO Runtime 之上**（Runtime 在独立仓库 [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)）；GenAI 内置基于 [openvino-tokenizers](https://github.com/openvinotoolkit/openvino-tokenizers) 的分词能力，"无需外部依赖即可运行生成式模型"（[README](https://github.com/openvinotoolkit/openvino.genai)）。
- OpenVINO Runtime 侧的官方设备口径为 **CPU、GPU、NPU** 三类，另有 AUTO（自动设备选择）、HETERO（异构切分）、BATCH（自动批处理）等推理模式；注意官方特性矩阵中 **NPU 不支持动态形状、不支持 Heterogeneous 执行、无多流执行**（[Supported Devices](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)）。
- plugin 源码位置：Runtime 仓库 `src/plugins/intel_cpu`、`intel_gpu`、`intel_npu`（[openvino 仓库](https://github.com/openvinotoolkit/openvino)）。

### 1.2 OVMS（OpenVINO Model Server）与 GenAI 的关系

- OVMS（[openvinotoolkit/model_server](https://github.com/openvinotoolkit/model_server)）是 Intel 官方模型服务组件。GenAI README 明确说明：**Continuous batching 功能被用于 OVMS 中以服务化 LLM**（含 prefix caching）。
- OVMS 提供 **OpenAI 兼容 API** 的 LLM 服务，内部使用 continuous batching + **paged attention**（[Continuous Batching demo 文档](https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching.html)、[LLM serving 参考页](https://docs.openvino.ai/2025/model-server/ovms_docs_llm_reference.html)），并支持 embeddings（OpenAI API）与 rerank（Cohere API）服务化。
- OVMS 的图像生成 demo 展示了扩散模型的**跨设备切分**：text encoder 与去噪（UNet）在 NPU、VAE decoder 在 GPU——"当模型太大放不进 NPU 内存时"的推荐部署方式（[OVMS image generation demo](https://docs.openvino.ai/2026/model-server/ovms_demos_image_generation.html)）。

### 1.3 llama.cpp 集成关系（两条线）

- **GGUF 读取（GenAI 侧）**：OpenVINO 2025.2 起，OpenVINO GenAI 增加了 **GGUF Reader**，Python/C++ 管线可直接加载 llama.cpp 格式的 GGUF 模型（[Intel 2025.2 Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-2.html)）。
- **OpenVINO 作为 llama.cpp 后端（方向相反）**：OpenVINO 后端已进入 llama.cpp 上游，可从 llama.cpp 主线在 Intel CPU/GPU/NPU 上跑 GGUF 模型（[llama.cpp OPENVINO.md 后端文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md)、[OpenVINO 官方博客 "OpenVINO™ Lands in llama.cpp"](https://medium.com/openvino-toolkit/openvino-lands-in-llama-cpp-run-gguf-models-on-intel-cpu-gpu-and-npu-d6fca1d633e8)）。OpenVINO 2026.1 起该后端进入预览/持续强化阶段（社区报道，[Igor's Lab](https://www.igorslab.de/en/intel-openvino-2026-1-links-llama-c-with-wildcat-lake-and-arc-pro-b70-suddenly-making-intels-ai-strategy-more-tangible/)；此条为第三方来源，细节以 llama.cpp 仓库为准）。

### 1.4 发布渠道与版本

| 渠道 | 说明 |
| --- | --- |
| PyPI | `pip install openvino-genai`（另有 `openvino`、`openvino-tokenizers`）；Node.js 为 npm 包 `openvino-genai-node`（[README](https://github.com/openvinotoolkit/openvino.genai)） |
| GitHub Releases | [openvino.genai/releases](https://github.com/openvinotoolkit/openvino.genai/releases)：2025 线为 2025.1.0.0 → 2025.4.1.0；2026 线已发布 2026.0.0.0 → **2026.3.0.0（最新，2026-08-05）** |
| 发行版 | 与 OpenVINO Runtime 同步：pip wheel、GitHub Archive、RPM、APT 等（[System Requirements/OpenVINO Distributions](https://docs.openvino.ai/systemrequirements)） |
| 预发布 | nightly wheel：`pip install --pre openvino openvino-tokenizers openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly`（[GenAI on NPU 指南](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)） |

2025 年各版本亮点（依据 [Releases](https://github.com/openvinotoolkit/openvino.genai/releases)、[OpenVINO 2025 Release Notes](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino.html) 及搜索结果）：2025.3 增加 Phi-4-mini-reasoning、Gemma-3-1B 等模型与 NPU 动态 prefill；**2025.4（2025-12-01）主打 Agentic AI（output parsing、改进 chat template）、更多模型在 CPU/GPU/NPU 上获得支持、更多压缩选项**。

---

## 2. 生成式模型管线支持（任务类型）

以下管线均出自 [openvino.genai README](https://github.com/openvinotoolkit/openvino.genai) 与 [官方 Supported Models 页面](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)（2026-09-06 核实）：

| 管线 / 任务 | GenAI API（Python/C++/Node.js） | 代表模型 | LoRA |
| --- | --- | --- | --- |
| LLM 文本生成 | `LLMPipeline`（TextGeneration） | Llama 2/3、Qwen 全系、Phi-3/4、Gemma 1–3、Mistral/Mixtral、ChatGLM/GLM-4、DeepSeek-V3/R1 等 | ✅（LLM 部分） |
| VLM 视觉语言 | `VLMPipeline`（支持图片/视频输入） | LLaVA/LLaVA-NeXT(-Video)、Qwen2/2.5/3-VL、MiniCPM-V/o、InternVL2/3、Phi-3-vision/Phi-4-multimodal、Gemma3/3n/4、VideoChat-Flash、DeepSeek-OCR-2 等 | ✅（仅 LLM 部分） |
| 语音识别 ASR | `WhisperPipeline`（SpeechRecognition） | Whisper 全系、Distil-Whisper、Qwen3-ASR、Fun-ASR-Nano | ❌ |
| 语音合成 TTS | TTS 管线（SpeechGeneration） | SpeechT5、Kokoro-82M（Kokoro 支持于 2026.3 加入，[PR #3483](https://github.com/openvinotoolkit/openvino.genai/releases)） | ❌ |
| 文生图 Text2Image | `Text2ImagePipeline`（Diffusers） | LCM、SD 1.x/2.x、SDXL、SD3/3.5、Flux.1、FLUX.2-klein、Qwen-Image、Z-Image | ✅（多数） |
| 图生图 / 重绘 Image2Image / Inpainting | 同上 | 同上（SD/SDXL/SD3/Flux 支持；FLUX.2/Z-Image/Qwen-Image 不支持 Inpainting） | ✅（多数） |
| **视频生成 Video Generation** | `Text2VideoPipeline` / Image2Video | **LTX-Video**（Text2Video ✅、Image2Video ✅；Text2Video 的 LoRA 支持于 2026.2 加入，[PR #3362](https://github.com/openvinotoolkit/openvino.genai/releases)） | ✅ |
| Embeddings（RAG） | `EmbeddingPipeline`（文本 + 多模态 text/image/video） | BGE、all-MiniLM、mxbai-embed、multilingual-e5、Qwen3-Embedding、Qwen3-VL-Embedding | ❌ |
| 文本 Rerank（RAG） | Rerank 管线 | bge-reranker-v2-m3、cross-encoder/ms-marco 系列、ModernBERT reranker、Qwen3-Reranker | ❌ |
| 服务化 Continuous Batching | GenAI 内置 CB 引擎（vLLM 风格：continuous batching + paged attention + prefix caching），经 OVMS 暴露 OpenAI 兼容 API | 任意受支持 LLM（[OVMS CB demo](https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching.html)） | — |

其他优化特性（[README](https://github.com/openvinotoolkit/openvino.genai)）：**投机解码（speculative decoding）**、**KVCache token eviction**、prefill **稀疏注意力（Tri-shape / XAttention）**、多 LoRA 适配器加载/混叠。OpenVINO 2025.4 增强了 Agentic 场景（结构化输出解析、chat template 改进）（[OpenVINO 2025.4 发布博客](https://medium.com/openvino-toolkit/openvino-2025-4-faster-models-smarter-agents-3709e6437a08)）。

> 说明：老版本 README 曾以 "Stable Video Diffusion" 示例展示视频能力，但当前官方 Supported Models 页面中视频生成一节仅正式列出 **LTX-Video** 架构；Stable Video Diffusion 未出现在当前支持表中。

---

## 3. 支持的模型清单（摘要）

完整清单见 [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)（含逐架构与 HuggingFace 示例模型）。要点：

- **LLM（约 60 个架构族）**：Llama 2/3.x（含 Llama 3.3 70B、DeepSeek-R1-Distill）、Qwen/Qwen2/2.5/Qwen3/MoE/Qwen3-Next（含 Qwen3.5/3.6/3.8）、Phi/Phi3/Phi-4(-reasoning)/Phi-3.5-MoE、Gemma/Gemma2/Gemma3、Mistral/Mixtral/Falcon3、ChatGLMModel（chatglm2/3、GLM-4）与 GlmForCausalLM（GLM-4-9B、glm-edge）、DeepSeek-MoE/V2/V3（含 DeepSeek-R1）、GPT-OSS、IBM Granite/GraniteMoE、Baichuan2、Bloom、InternLM/2、MiniCPM 1–4、Exaone 3.5/4.0、LFM2/LFM2MoE、Aya/Command R（Cohere）、StarCoder/2、Neural Chat、OLMo、MPT、OPT、XGLM、XVERSE、Aquila 等。
- **VLM**：LLaVA 1.5 / NeXT / NeXT-Video / nanoLLaVA、MiniCPM-V-2.6 / MiniCPM-o-2.6、Qwen2-VL / Qwen2.5-VL / Qwen3-VL（含 Qwen3.5/3.6/3.8 多模态分支）、InternVL2/2.5/3、Phi-3-vision / Phi-3.5-vision / Phi-4-multimodal、Gemma3 / 3n / 4（text+image+video）、VideoChat-Flash、Fara-7B、DeepSeek-OCR-2、Muse-Glimmer 等。
- **图像生成**：LCM、SD 1.x–2.x、SD Inpainting、SDXL(+Inpainting)、SD3/3.5 medium/large(+Turbo)、Flux.1（schnell 等）、FLUX.2 [klein]、Qwen-Image、Z-Image。
- **视频生成**：LTX-Video。
- **语音**：Whisper（tiny–large-v3）、Distil-Whisper、Qwen3-ASR、Fun-ASR-Nano；TTS：SpeechT5、Kokoro-82M。
- **Embedding / Rerank**：BGE 系列、all-MiniLM/mpnet/distilroberta、e5、mxbai、Qwen3-Embedding、Qwen3-VL-Embedding（多模态）；rerank：ms-marco cross-encoder、bge-reranker-v2-m3、gte-reranker-modernbert、Qwen3-Reranker。
- **转换通道（Optimum-Intel / OVConverter）**：GenAI 的模型消费入口是 [Optimum-Intel](https://github.com/huggingface/optimum-intel) 的 `optimum-cli export openvino`（导出 OpenVINO IR，LLM 默认带 KV-cache 的 stateful 模型）。Optimum-Intel 支持的架构表（[官方文档](https://huggingface.co/docs/optimum-intel/openvino/models)）比 GenAI 验证清单更宽（含 SAM、T5、MarianMT、ViT/ResNet 等通用任务），GenAI 管线则要求模型签名匹配（LLM 需要 `input_ids / attention_mask / beam_idx / position_ids(可选)` + 单 logits 输出，见 [Supported Models 的 info 注记](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)）。官方同时提示：未列出但架构相似的模型也可能可用，需自行验证。

---

## 4. Intel 硬件支持矩阵（设备 × 模型类型）

### 4.1 官方支持的设备清单

依据 [OpenVINO System Requirements](https://docs.openvino.ai/systemrequirements)：

- **CPU**：Intel Core Ultra Series 1/2/3、Xeon 6、6–14 代酷睿、1–5 代 Xeon Scalable、Atom 系列；另有 ARM64（Apple silicon 等）支持（Windows ARM64 不支持 CPU 推理）。CPU plugin 覆盖 x86-64 与 ARM，原生 FP32/BF16/FP16/INT8/MXFP4（MXFP4 仅 x86-64），AMX 矩阵扩展在 4 代 Xeon Scalable（ Sapphire Rapids）及以上由 bf16/f16 路径激活（[CPU Device 文档](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html)）。
- **GPU**：Intel Arc GPU 系列（含 Arc iGPU 与 Arc dGPU）、HD/UHD Graphics、Iris Pro / Iris Xe / Iris Xe Max、**Data Center GPU Flex / Max 系列**。GPU plugin 基于 OpenCL，支持 iGPU 与 dGPU，多 tile（如 Max 系列）可用 `GPU.x.y` 寻址（[GPU Device 文档](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html)）。
- **NPU**：随 Intel Core Ultra（Meteor Lake 起）引入的 NPU（**NPU 3720**），Host 平台为 Core Ultra 系列；需要单独安装 NPU 驱动，支持 Windows 11 与 Ubuntu 22.04/24.04（[NPU Device 文档](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)）。
- **Gaudi**：**不在 OpenVINO 设备列表内**（见 [§7.2](#72-intel-gaudi--不是-openvino-设备)）。

### 4.2 核心矩阵

| 模型类型 / 管线 | Intel CPU（Core/Xeon，AVX2/AVX-512/AMX） | Intel iGPU（Core Ultra Arc iGPU / Iris Xe / UHD） | Intel Arc dGPU / Data Center GPU Flex·Max | Intel NPU（Core Ultra 3720） | 依据 |
| --- | --- | --- | --- | --- | --- |
| **LLM（TextGeneration）** | ✅ 全面支持（INT4/INT8 权重压缩、投机解码、连续批处理） | ✅ 支持（prefix caching、投机解码；KV cache INT8 等优化） | ✅ 支持（多 GPU / CUMULATIVE_THROUGHPUT；OVMS 多卡 CB） | ✅ 支持，但有量化与形状硬约束：须对称 INT4/NF4（`--sym --weight-format int4/nf4`，`--ratio 1.0`）；NF4 仅 Core Ultra Series 2+；默认静态形状，`MAX_PROMPT_LEN=1024`/`MIN_RESPONSE_LEN=128`，2025.3 起支持动态 prefill 分块 | [GenAI on NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)、[README](https://github.com/openvinotoolkit/openvino.genai) |
| **VLM（视觉语言）** | ✅ | ✅ | ✅ | ✅（与 LLM 同样受 NPU 限制约束；参数经 `DEVICE_PROPERTIES.NPU` 传入） | [GenAI on NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)、[Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/) |
| **语音识别（Whisper 等）** | ✅ | ✅ | ✅ | ✅ 无 NPU 专属限制；2024.5 首次支持 NPU，2025.1 起 stateful Whisper 直接可用（FP16/INT8 均可）；需较新 NPU 驱动（排障建议 ≥32.0.100.3104） | [GenAI on NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html) |
| **TTS（SpeechT5 / Kokoro）** | ✅ | ✅ | ✅ | ⚠️ 官方文档未标注 NPU 支持（未在 NPU 指南出现） | [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/) |
| **文生图（SD/SDXL/SD3.5/Flux/Qwen-Image/Z-Image）** | ✅（可用，速度受限） | ✅（iGPU 可跑，SDXL/SD3.5 建议高显存平台） | ✅ **推荐设备**（显存充足、性能最佳） | ⚠️ 部分/分段支持：OVMS 官方 demo 为 text encoder + denoising 在 NPU、**VAE decoder 放 GPU**（模型放不进 NPU 内存时）；未见"完整 SD 管线纯 NPU"的官方端到端承诺 | [OVMS image generation demo](https://docs.openvino.ai/2026/model-server/ovms_demos_image_generation.html) |
| **图生图 / Inpainting** | ✅ | ✅ | ✅ 推荐 | ⚠️ 同文生图（按组件切分） | 同上 |
| **视频生成（LTX-Video）** | ✅（可用） | ✅ | ✅（推荐） | ❌ 官方文档未标注 NPU 支持 | [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)、[2026.2 release notes](https://github.com/openvinotoolkit/openvino.genai/releases) |
| **Embeddings（RAG）** | ✅ | ✅ | ✅ | ⚠️ 有 NPU embedding 迹象（2026.3 代码中出现 `test_qwen3_embedding_npu`），但官方 NPU 指南未系统描述，视为部分/实验性 | [Releases](https://github.com/openvinotoolkit/openvino.genai/releases) |
| **Rerank（RAG）** | ✅ | ✅ | ✅ | ❌ 官方文档未标注 NPU 支持 | [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/) |
| **Continuous Batching 服务化（OVMS）** | ✅ | ✅ | ✅（多 GPU） | ❌ 不适用（CB/paged attention 面向 CPU/GPU 服务端） | [OVMS CB demo](https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching.html) |
| **Intel Gaudi 2/3** | — | — | — | — | ❌ **不支持**：走 SynapseAI + vLLM 插件路线（[§7.2](#72-intel-gaudi--不是-openvino-设备)） |

> 图例：✅ 官方明确支持；⚠️ 部分/有条件/文档未明确；❌ 不支持或未标注。

---

## 5. 设备特定限制与注意事项

### 5.1 NPU（重点）

来自 [OpenVINO GenAI on NPU 指南](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html) 与 [NPU Device 文档](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)：

1. **仅支持静态形状模型**（NPU Device 官方 Limitations）。GenAI 的 NPU LLM 管线基于静态形状思路：默认 `MAX_PROMPT_LEN=1024`、`MIN_RESPONSE_LEN=128`；OpenVINO 2025.3 起引入动态 prompt 支持（`NPUW_LLM_PREFILL_CHUNK_SIZE`，默认 1024 分块）。
2. **量化硬约束**：LLM 必须以**对称 4-bit**（INT4 或 NF4）导出（`--sym`、`--weight-format int4/nf4`、`--ratio 1.0`）；channel-wise（`--group-size -1`）性能通常最好但可能伤精度（可用 NNCF 的 AWQ/Scale Estimation/GPTQ 补偿）；group-wise（`--group-size 128`）适合 ≤4–5B 小模型。**NF4 仅 Core Ultra Series 2（Lunar Lake）及以上 NPU 支持**。
3. **内存门槛**：Core Ultra Series 2 平台上，>7B 模型（Llama-2-7B、Qwen-2-7B 等）处理 >1024 token 的 prompt 可能需要 **16GB 以上系统内存**。
4. **算子/子图兜底机制**：NPU plugin 无法执行的算子由 NPUW（NPU wrapper）层处理，实际部署中存在"静默落到不支持的配置而不回退"的风险（社区 issue [openvino#35641](https://github.com/openvinotoolkit/openvino/issues/35641) 报告 `LLMPipeline(..., "NPU")` 接受不支持的 INT8 配置后崩溃且无回退诊断）。这一点官方文档未系统描述，属于需要注意的工程风险。
5. **编译与缓存**：NPU 编译耗时明显，官方提供 `CACHE_DIR`（OpenVINO 缓存，2025.1 起推荐）、ahead-of-time blob 导出（`EXPORT_BLOB/BLOB_PATH`，含加密回调）；blob 跨 OpenVINO/驱动版本不保证兼容，不建议生产环境使用预编译 blob（[NPU Device](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)）。
6. **驱动要求**：需安装 NPU 驱动；执行失败时官方建议升级驱动至 **32.0.100.3104+**，OOM 时可设 `DISABLE_OPENVINO_GENAI_NPU_L0=1` 关闭 Level0 内存分配。
7. **特性矩阵短板**（[Supported Devices](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)）：NPU 不支持动态形状、Heterogeneous 执行、多流、自动批处理、预处理加速与扩展性（extensibility）；模型缓存为 Partial。

### 5.2 GPU

- iGPU（`GPU.0` 恒为集成显卡）与 dGPU 混布时可用 `AUTO:GPU.1,GPU.0` + CUMULATIVE_THROUGHPUT 多卡并行；Max 系列多 tile 支持 `GPU.x.y`（[GPU Device](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html)）。
- 动态形状为 preview 特性（主要面向 NLP 模型），离散 GPU 上可能劣于静态形状；建议优先静态形状 + 模型缓存。
- INT8/u8 硬件加速需要 Iris Xe / Xe MAX 及以上代际；老平台回退浮点执行。

### 5.3 CPU

- 4 代 Xeon Scalable 及以上通过 BF16 路径启用 **AMX**，显著快于 AVX-512/AVX2；无 AVX-512_BF16 的 AVX-512 平台有 bf16 软件仿真模式（仅开发用途）。INT8 为 x86-64 原生；ARM 平台量化模型以浮点模拟执行；Windows ARM64 不支持（[CPU Device](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html)）。
- BF16 精度可能影响 LLM 精度，可用 `ov::hint::execution_mode=ACCURACY` 强制 FP32。

---

## 6. 量化支持（NNCF 权重压缩 / KV cache / 连续批处理）

- **权重压缩（NNCF）**：Optimum-Intel 底层用 NNCF 做 PTQ——INT8 静态量化（校准集）与 **INT4/NF4 权重压缩**（`--weight-format int4/nf4`、`--group-size`、`--ratio`、`--sym`），并支持数据感知方法 **AWQ、Scale Estimation、GPTQ**（[Optimum-Intel](https://github.com/huggingface/optimum-intel)、[GenAI on NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)）。CPU 额外支持 **MXFP4** 推理精度（[CPU Device](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html)）。
- **按设备的量化要求**：CPU/GPU 对 INT4/INT8/FP16 均可用；**NPU 只吃对称 4-bit（INT4-CW/GQ 或 NF4-CW）**，NPU 侧还有 `compiler_dynamic_quantization`、`qdq_optimization` 等专属编译属性（[NPU Device](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)）。
- **KV cache**：GenAI 管线消费 stateful（带 KV cache）的 IR 模型（Whisper 自 2025.1 起在 NPU 上也支持 stateful）；LLM/VLM/Whisper 的 KV cache 管理、token eviction 由 GenAI 负责；NPU 上下文长度由 `MAX_PROMPT_LEN + MIN_RESPONSE_LEN` 决定（KV cache 显式静态分配）（[GenAI on NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)）。OVMS 连续批处理引擎实现 **paged attention** 并支持 prefix caching（[OVMS CB demo](https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching.html)）。
- **连续批处理**：GenAI 内置 continuous batching（vLLM 风格）+ prefix caching，README 明确其用途是"在 OVMS 中服务 LLM"；NPU 不在 CB 服务化场景中（见 §4.2）。
- **投机解码与稀疏注意力**：README 列为跨管线优化；2025.4 起继续强化，且搜索结果表明投机解码在 2026.x 已支持 NPU（[OpenVINO 2025.4 博客](https://medium.com/openvino-toolkit/openvino-2025-4-faster-models-smarter-agents-3709e6437a08)及 [GenAI on NPU 文档](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)上下文；后者未逐条展开，属中等置信度）。

---

## 7. 与相邻技术栈的关系

### 7.1 IPEX-LLM（对比参考）

- [intel/ipex-llm](https://github.com/intel/ipex-llm)（前身 BigDL-LLM，2024-03 更名）是 PyTorch 生态的 Intel 客户端 LLM 加速库（基于 IPEX/SYCL，主攻 Arc GPU/CPU/NPU）。
- **该仓库已于 2026-01-28 归档（只读，不再接受 Intel 补丁）**；Intel 同时宣布 BigDL 停止开发（社区报道）。Intel 本地 LLM 路线事实上收敛到 **OpenVINO GenAI / OVMS**，另有一些新项目（如社区所称 "llm-scaler"，官方口径不清，此处存疑）。
- 分工总结：IPEX-LLM = PyTorch/HF transformers 路线的低门槛加速；OpenVINO GenAI = IR/图表示路线，跨 CPU/GPU/NPU 统一，官方主推。

### 7.2 Intel Gaudi —— 不是 OpenVINO 设备

- OpenVINO 官方设备列表只有 CPU/GPU/NPU（[Supported Devices](https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html)），**不存在 Gaudi device plugin**。
- Gaudi 2/3 的 LLM 推理路线是：**SynapseAI（Habana 驱动/固件/运行时，Gaudi 文档现版本 1.24.x）+ vLLM Hardware Plugin for Intel Gaudi**（Intel 自维护，仓库 [HabanaAI/vllm-fork](https://github.com/HabanaAI/vllm-fork)，官方文档 [docs.vllm.ai/projects/gaudi](https://docs.vllm.ai/projects/gaudi/en/latest/general/faq.html)）+ 训练/迁移用 Optimum-Habana。SynapseAI 1.24.0 兼容 vLLM 插件 0.17.1–0.21.0（[Gaudi Release Notes](https://docs.habana.ai/en/latest/Release_Notes/GAUDI_Release_Notes.html)）。
- Gaudi 上的官方性能数据基于 Optimum-Habana / SynapseAI（[Intel Gaudi 3 Model Performance](https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance-1-19.html)）。社区部署资料还指出 vLLM on Gaudi 3 目前仅 BF16 精度受支持（Dell 部署指南，第三方来源）。
- 结论：**若目标是 Gaudi，应使用 vLLM（Gaudi 插件）/ Optimum-Habana，而不是 openvino.genai**；OpenVINO 在 Intel 数据中心的角色落在 Xeon CPU（含 AMX）与 Data Center GPU Max/Flex 上。

### 7.3 OVMS

见 [§1.2](#12-ovmsopenvino-model-server-与-genai-的关系)：GenAI 是引擎，OVMS 是服务壳（OpenAI 兼容 API + CB/paged attention + embeddings/rerank/图像生成服务）。

### 7.4 llama.cpp

见 [§1.3](#13-llamacpp-集成关系两条线)：GenAI 侧 GGUF Reader（2025.2+）+ llama.cpp 上游 OpenVINO backend（反向集成）。

---

## 8. 参考来源列表

**一手来源（官方）**
1. OpenVINO GenAI 仓库与 README — https://github.com/openvinotoolkit/openvino.genai
2. OpenVINO GenAI Supported Models — https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/
3. OpenVINO GenAI Releases — https://github.com/openvinotoolkit/openvino.genai/releases
4. OpenVINO Runtime 仓库（plugins）— https://github.com/openvinotoolkit/openvino
5. OpenVINO Supported Devices（2025 文档）— https://docs.openvino.ai/2025/documentation/compatibility-and-support/supported-devices.html
6. OpenVINO System Requirements — https://docs.openvino.ai/systemrequirements
7. CPU / GPU / NPU Device 文档（2025）— https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html 、…/gpu-device.html 、…/npu-device.html
8. OpenVINO GenAI on NPU 指南（2026 文档）— https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html
9. OVMS Continuous Batching demo — https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching.html ；LLM serving 参考 — https://docs.openvino.ai/2025/model-server/ovms_docs_llm_reference.html ；model_server 仓库 — https://github.com/openvinotoolkit/model_server
10. OVMS 图像生成 demo（NPU+GPU 切分）— https://docs.openvino.ai/2026/model-server/ovms_demos_image_generation.html
11. Intel OpenVINO 2025.2 Release Notes（GGUF Reader）— https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-2.html ；2025 Release Notes — https://docs.openvino.ai/2025/about-openvino/release-notes-openvino.html
12. Optimum-Intel 仓库与支持模型表 — https://github.com/huggingface/optimum-intel 、https://huggingface.co/docs/optimum-intel/openvino/models
13. OpenVINO 官方博客：OpenVINO Lands in llama.cpp — https://medium.com/openvino-toolkit/openvino-lands-in-llama-cpp-run-gguf-models-on-intel-cpu-gpu-and-npu-d6fca1d633e8 ；OpenVINO 2025.4 博客 — https://medium.com/openvino-toolkit/openvino-2025-4-faster-models-smarter-agents-3709e6437a08

**Gaudi / 相邻生态**
14. vLLM Hardware Plugin for Intel Gaudi FAQ — https://docs.vllm.ai/projects/gaudi/en/latest/general/faq.html
15. Habana/Gaudi Release Notes（SynapseAI 1.24）— https://docs.habana.ai/en/latest/Release_Notes/GAUDI_Release_Notes.html
16. HabanaAI vllm-fork — https://github.com/HabanaAI/vllm-fork
17. Intel Gaudi 3 Model Performance（SynapseAI 1.19）— https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance-1-19.html
18. intel/ipex-llm（已归档 2026-01-28）— https://github.com/intel/ipex-llm

**其他**
19. llama.cpp OpenVINO 后端文档 — https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md
20. openvino#35641（NPU 静默不回退问题）— https://github.com/openvinotoolkit/openvino/issues/35641
21. Igor's Lab 对 OpenVINO 2026.1 / llama.cpp 后端的报道（第三方）— https://www.igorslab.de/en/intel-openvino-2026-1-links-llama-c-with-wildcat-lake-and-arc-pro-b70-suddenly-making-intels-ai-strategy-more-tangible/

### 来源冲突与不确定点（明示）

- **2025.4.1.0 的具体发布日期**：搜索结果出现 "Aug 26" 字样但与 2025.4（2025-12-01）时间线矛盾，疑似搜索摘要误读；本报告仅依据 [Releases 页](https://github.com/openvinotoolkit/openvino.genai/releases)确认其在 2026.0.0.0 之前发布，不给出精确日期。
- **NPU 上 Embedding / TTS / 视频生成支持**：官方 NPU 指南未系统描述；Embedding 在 NPU 有代码级迹象（`test_qwen3_embedding_npu`），本报告标为"部分/未标注"。
- **投机解码在 NPU 的支持**：来自搜索摘要对 2026.x 的转述，NPU 指南正文未逐条确认，标注为中等置信度。
- **OpenVINO 2025 文档 vs 2026 文档**：本报告混用了 2025（设备/插件细节）与 2026（GenAI on NPU）两版文档；两版在设备能力大方向上一致，NPU 静态形状限制等以 2025 版特性矩阵为准。
