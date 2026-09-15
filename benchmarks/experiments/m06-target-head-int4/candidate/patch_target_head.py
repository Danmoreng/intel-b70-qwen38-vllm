"""Install a default-off INT4/G128 target-head arm for TP1 qualification."""
from pathlib import Path


path = Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5.py")
source = path.read_text()
old = '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''
new = '''    def _b70_target_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if os.environ.get("B70_TARGET_LMHEAD_INT4") != "1":
            return None
        packed = getattr(self, "_b70_target_lmhead_int4", None)
        if packed is None:
            return None
        from vllm.model_executor.models.b70_draft_lmhead_int4 import int4_lmhead_logits
        return int4_lmhead_logits(hidden_states, *packed)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self._b70_target_logits(hidden_states)
        if logits is not None:
            return logits
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._b70_target_logits(hidden_states)
        if logits is not None:
            return logits
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        if os.environ.get("B70_TARGET_LMHEAD_INT4") == "1":
            from vllm.model_executor.models.b70_draft_lmhead_int4 import quantize_lmhead_to_int4
            if self.vllm_config.parallel_config.tensor_parallel_size != 1:
                raise RuntimeError("B70 target INT4 qualification is TP1-only")
            source_weight = self.lm_head.weight
            source_shape = tuple(source_weight.shape)
            self._b70_target_lmhead_int4 = quantize_lmhead_to_int4(source_weight.detach())
            torch.xpu.synchronize()
            if os.environ.get("B70_TARGET_LMHEAD_FREE_FP16") == "1":
                # Qwen3.5 has tie_word_embeddings=false and Step3.5 explicitly
                # keeps a separate draft lm_head. Both compute_logits paths
                # above fail closed to the packed tensor before releasing this
                # independent target-only source allocation.
                self.lm_head.weight = torch.nn.Parameter(
                    source_weight.new_empty(0), requires_grad=False)
            print("B70_TARGET_LMHEAD_INT4_READY", source_shape,
                  "source_numel", self.lm_head.weight.numel(), flush=True)
        return loaded
'''
assert source.count(old) == 1, source.count(old)
if "\nimport os\n" not in source:
    source = source.replace("import torch\n", "import os\nimport torch\n", 1)
path.write_text(source.replace(old, new, 1))
