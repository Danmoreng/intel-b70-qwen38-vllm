"""Install a default-off INT4/G128 target-head arm for TP1 qualification."""
from pathlib import Path


path = Path("/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5.py")
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

    def _b70_finalize_target_head(self) -> None:
        # AutoWeightsLoader may visit the language model more than once as the
        # checkpoint prefixes alternate. Only the OUTER load_weights may call
        # this, after it has exhausted the complete checkpoint iterator.
        if os.environ.get("B70_TARGET_LMHEAD_INT4") == "1":
            from vllm.model_executor.models.b70_draft_lmhead_int4 import quantize_lmhead_to_int4
            parallel = self.vllm_config.parallel_config
            if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
                raise RuntimeError("B70 target INT4 qualification requires TP1/PP1")
            if self.config.tie_word_embeddings:
                raise RuntimeError("B70 target INT4 requires an independent output head")
            if getattr(self, "_b70_target_lmhead_int4", None) is not None:
                raise RuntimeError("B70 target INT4 does not support weight reload")
            source_weight = self.lm_head.weight
            if tuple(source_weight.shape) != (self.config.vocab_size, self.config.hidden_size):
                raise RuntimeError("B70 target INT4 head shape does not match the model")
            source_shape = tuple(source_weight.shape)
            self._b70_target_lmhead_int4 = quantize_lmhead_to_int4(source_weight.detach())
            # The executing V2 Eagle/MTP loader shares this module with the
            # draft. Carry the packed weights on that exact shared object.
            self.lm_head._b70_target_lmhead_int4 = self._b70_target_lmhead_int4
            torch.xpu.synchronize()
            if os.environ.get("B70_TARGET_LMHEAD_FREE_FP16") == "1":
                # Both target logits routes use the packed head, and the draft
                # builder below adopts it before touching the released source.
                self.lm_head.weight = torch.nn.Parameter(
                    source_weight.new_empty(0), requires_grad=False)
            print("B70_TARGET_LMHEAD_INT4_READY", source_shape,
                  "source_numel", self.lm_head.weight.numel(), flush=True)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if getattr(self, "_b70_target_lmhead_int4", None) is not None:
            raise RuntimeError("B70 target INT4 does not support weight reload")
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''
outer_old = '''    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits_local(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''
outer_new = '''    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits_local(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        if os.environ.get("B70_TARGET_LMHEAD_INT4") == "1":
            if "language_model.lm_head.weight" not in loaded:
                raise RuntimeError("B70 target INT4 requires a loaded output head")
            self.language_model._b70_finalize_target_head()
        return loaded
'''


def patch_source(source: str) -> str:
    assert source.count(old) == 1, source.count(old)
    assert source.count(outer_old) == 1, source.count(outer_old)
    if "\nimport os\n" not in source:
        source = source.replace("import torch\n", "import os\nimport torch\n", 1)
    result = source.replace(old, new, 1).replace(outer_old, outer_new, 1)
    compile(result, str(path), "exec")
    return result


def patch_draft_source(source: str) -> str:
    old_head = '    head = getattr(model, "lm_head", None)\n    weight = getattr(head, "weight", None)\n'
    new_head = '''    head = getattr(model, "lm_head", None)
    packed = getattr(head, "_b70_target_lmhead_int4", None)
    if packed is not None:
        model._b70_lmhead_int4 = packed
        print("B70_DRAFT_LMHEAD_INT4_REUSED_TARGET", flush=True)
        return
    weight = getattr(head, "weight", None)
'''
    assert source.count(old_head) == 1, source.count(old_head)
    result = source.replace(old_head, new_head, 1)
    compile(result, "b70_draft_lmhead_int4.py", "exec")
    return result


if __name__ == "__main__":
    draft_path = path.with_name("b70_draft_lmhead_int4.py")
    target = patch_source(path.read_text())
    draft = patch_draft_source(draft_path.read_text())
    path.write_text(target)
    draft_path.write_text(draft)
