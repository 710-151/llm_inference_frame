# mini-vllm

轻量级 LLM 推理框架，支持 paged KV cache、GQA、RoPE、RMSNorm、连续批处理。

## 支持的模型

| 模型族 | model_type | 实现方式 | 备注 |
|--------|------------|----------|------|
| LLaMA-3.2-1B | `llama` | config 驱动 | |
| Qwen2-0.5B | `qwen2` | config 驱动 | |
| Qwen2.5-0.5B | `qwen2.5` | config 驱动 | |
| Qwen3-0.6B | `qwen3` | config 驱动 | QK-norm |
| Qwen3 MoE(minimind3-moe) | `qwen3_moe` | 独立实现 | 稀疏 MoE，top-k 路由，QK-norm |
| Qwen 3.5-0.8B | `qwen3_5_text` | 独立实现 | 混合注意力（Gated Delta Net 线性 + 全注意力），三层状态管理 |
| Gemma 3-1B | `gemma3_text` | 独立实现 | per-layer RoPE, feedforward layernorms |

modelscope社区下载对于的模型权重文件 ，所有模型均支持 Instruct 版本，不支持量化版本，自动应用对应的 chat template。

## 安装

```bash
pip install .
```

开发依赖：

```bash
pip install -e ".[dev]"
```

## 快速开始

### 离线推理

```python
from mini_vllm.engine.engine import LLMEngine
from mini_vllm.utils.config import EngineConfig, SamplingParams

config = EngineConfig(
    model_path="/path/to/model",  # HuggingFace 格式模型路径
    device="cuda",                 # 或 "cpu"
    dtype="float16",
)
engine = LLMEngine(config)

# 使用 chat template 格式化输入（推荐，适用于 Instruct 模型）
messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]
prompt = engine.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

results = engine.generate([prompt], SamplingParams(temperature=0.7, max_tokens=256))
print(results[0].output_text)
```

### 流式推理

```python
for idx, token_text, finished in engine.generate_stream([prompt], sampling_params):
    print(token_text, end="", flush=True)
```

### 启动 API 服务

```bash
python -m mini_vllm.server.api_server --model-path /path/to/model \
    --device cuda \
    --dtype float16 \
    --host 0.0.0.0 \
    --port 8000
    
python -m mini_vllm.server.api_server --model-path D:\models\gongjy\minimind-3-moe
python -m mini_vllm.server.api_server --model-path D:\models\Qwen\Qwen2-0.5B-Instruct
python -m mini_vllm.server.api_server --model-path D:\models\Qwen\Qwen2.5-0.5B-Instruct
python -m mini_vllm.server.api_server --model-path D:\models\Qwen\Qwen3-0.6B
python -m mini_vllm.server.api_server --model-path D:\models\Qwen\Qwen3.5-0.8B
python -m mini_vllm.server.api_server --model-path D:\models\LLM-Research\Llama-3.2-1B-Instruct
python -m mini_vllm.server.api_server --model-path D:\models\LLM-Research\gemma-3-1b-it
```

### 使用 OpenAI 客户端调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Chat Completion（自动使用模型的 chat template）
response = client.chat.completions.create(
    model="model-name",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "什么是快速排序？"},
    ],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)

# 流式输出
stream = client.chat.completions.create(
    model="model-name",
    messages=[{"role": "user", "content": "用Python写一个快速排序"}],
    stream=True,
    max_tokens=256,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## 配置参数

### EngineConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | (必填) | HuggingFace 格式模型路径 |
| `device` | `"cuda"` | 推理设备，支持 `"cuda"` / `"cpu"` |
| `dtype` | `"auto"` | 数据类型，`"auto"` 会根据设备自动选择 |
| `block_size` | `16` | KV cache block 大小 |
| `max_num_seqs` | `256` | 最大并发序列数 |
| `max_num_batched_tokens` | `8192` | 最大批处理 token 数 |
| `gpu_memory_utilization` | `0.9` | GPU 显存使用比例 |

### SamplingParams

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | `1.0` | 采样温度，`0` 为贪心解码 |
| `top_k` | `-1` | top-k 采样，`-1` 表示不启用 |
| `top_p` | `1.0` | nucleus 采样 |
| `max_tokens` | `512` | 最大生成 token 数 |
| `stop` | `[]` | 停止字符串列表 |
| `presence_penalty` | `0.0` | 存在惩罚 |

## API 接口

### GET /v1/models

返回可用模型列表。

### POST /v1/chat/completions

Chat 格式补全，兼容 OpenAI Chat API。支持流式（`stream: true`）。

### POST /v1/completions

文本补全接口。支持流式。

## 项目结构

```
mini_vllm/
├── engine/
│   ├── engine.py           # LLMEngine 核心引擎（prefill + decode + 流式）
│   ├── kv_cache.py         # Paged KV cache 实现
│   ├── sampler.py          # 采样策略（greedy / temperature / top-k / top-p）
│   ├── scheduler.py        # 连续批处理调度
│   └── sequence.py         # 序列状态管理（多 EOS + stop 字符串检测）
├── models/
│   ├── model.py            # 通用 transformer 基类（RMSNorm, RoPE, GQA Attention）  [共用]
│   ├── standard_models.py  # LLaMA, Qwen2 / 2.5 / 3（config 驱动）                  [共用]
│   ├── gemma3_model.py     # Gemma 3 独立实现                                        [共用]
│   ├── qwen3_5_model.py    # Qwen 3.5 混合注意力独立实现（GatedDeltaNet + 全注意力）   [共用]
│   ├── qwen3_moe_model.py  # Qwen3 MoE 稀疏专家模型                                  [本次新增]
│   └── loader.py           # 模型加载、权重自动检测、model_type 注册                    [本次改动]
├── server/
│   ├── api_server.py       # OpenAI 兼容 API 服务（自动 chat template）
│   └── openai_types.py     # API 类型定义
├── tokenizer/
│   └── tokenizer.py        # Tokenizer 封装（encode / decode / apply_chat_template）
└── utils/
    ├── config.py           # EngineConfig + SamplingParams
    └── logger.py           # 日志
```

## 测试

```bash
# 单元测试
python -m pytest tests/test_config.py tests/test_sequence.py tests/test_engine.py tests/test_sampler.py tests/test_kv_cache.py tests/test_scheduler.py tests/test_model.py tests/test_model_family.py tests/test_qwen3_moe.py -v

# E2E 测试（需要下载模型，默认路径在测试文件中配置）
python -m pytest tests/test_e2e_real_model.py -v
python -m pytest tests/test_e2e_all_models.py -v
```
