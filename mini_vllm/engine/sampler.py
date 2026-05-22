import torch
from mini_vllm.utils.config import SamplingParams


class Sampler:
    def sample(self, logits: torch.Tensor, sampling_params: SamplingParams) -> list[int]:
        """
        Args:
            logits: [batch_size, vocab_size]
            sampling_params: sampling parameters
        Returns:
            sampled token_id for each sequence
        """
        if sampling_params.temperature == 0.0:
            return self._greedy(logits)
        return self._sample_with_temperature(logits, sampling_params)

    def _greedy(self, logits: torch.Tensor) -> list[int]:
        return logits.argmax(dim=-1).tolist()

    def _sample_with_temperature(self, logits: torch.Tensor, params: SamplingParams) -> list[int]:
        logits = logits / params.temperature

        if params.top_k > 0:
            logits = self._top_k_filter(logits, params.top_k)

        if params.top_p < 1.0:
            logits = self._top_p_filter(logits, params.top_p)

        probs = torch.softmax(logits, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1)
        return token_ids.squeeze(-1).tolist()

    def _top_k_filter(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
        return logits

    def _top_p_filter(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")

        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)
        return logits
