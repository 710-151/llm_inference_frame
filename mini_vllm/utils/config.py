from dataclasses import dataclass, field
import torch


@dataclass
class EngineConfig:
    model_path: str
    device: str = "cuda"
    dtype: str = "auto"
    block_size: int = 16
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.9
    tp_size: int = 1

    def get_torch_dtype(self) -> str:
        if self.dtype != "auto":
            return self.dtype
        return "float32" if self.device == "cpu" else "float16"

    def get_num_gpu_blocks(self, head_dim: int, num_layers: int, num_heads: int) -> int:
        """估算可用的 KV Cache block 数量"""
        if self.device == "cpu":
            return 256
        free_memory = torch.cuda.mem_get_info()[0]
        usable_memory = free_memory * self.gpu_memory_utilization
        dtype_size = 2 if self.get_torch_dtype() == "float16" else 4
        block_memory = 2 * self.block_size * num_heads * head_dim * num_layers * dtype_size
        return max(int(usable_memory / block_memory), 1)


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    presence_penalty: float = 0.0
    eos_token_id: int | list[int] = 2  # Overridden by engine from model config
