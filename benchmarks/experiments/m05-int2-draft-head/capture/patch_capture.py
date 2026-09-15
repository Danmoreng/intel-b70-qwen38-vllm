from pathlib import Path


site = Path("/opt/venv/lib/python3.12/site-packages/vllm")

# The V2 target path calls LogitsProcessor directly instead of the model's
# compute_logits wrapper. Hook the common projection point.
target = site / "model_executor/layers/logits_processor.py"
source = target.read_text()
old = '''        """Project hidden states through the lm_head, honoring head_dtype."""
        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
'''
new = '''        """Project hidden states through the lm_head, honoring head_dtype."""
        from b70_capture_hidden import capture
        capture("target", hidden_states)
        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
'''
assert source.count(old) == 1, source.count(old)
target.write_text(source.replace(old, new, 1))

# The deployed draft INT4 helper bypasses LogitsProcessor._apply_head.
draft = site / "model_executor/models/b70_draft_lmhead_int4.py"
source = draft.read_text()
old = '''def draft_lmhead_int4_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """Logits del draft via la copia INT4 (4 pasadas/paso -> 0.66 GB c/u)."""
    qweight, scales, qzeros, group_size = model._b70_lmhead_int4
'''
new = '''def draft_lmhead_int4_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """Logits del draft via la copia INT4 (4 pasadas/paso -> 0.66 GB c/u)."""
    from b70_capture_hidden import capture
    capture("draft", hidden_states)
    qweight, scales, qzeros, group_size = model._b70_lmhead_int4
'''
assert source.count(old) == 1, source.count(old)
draft.write_text(source.replace(old, new, 1))

# Capture the actual runtime tensors outside compiled model executables.
runner = site / "v1/worker/gpu/model_runner.py"
source = runner.read_text()
old = '''        else:
            sample_hidden_states = hidden_states[input_batch.logits_indices]
            logits = self.model.compute_logits(sample_hidden_states)
'''
new = '''        else:
            sample_hidden_states = hidden_states[input_batch.logits_indices]
            from b70_capture_hidden import capture
            capture("target", sample_hidden_states)
            logits = self.model.compute_logits(sample_hidden_states)
'''
assert source.count(old) == 1, source.count(old)
runner.write_text(source.replace(old, new, 1))

speculator = site / "v1/worker/gpu/spec_decode/speculator.py"
source = speculator.read_text()
old = '''    def _greedy_sample_draft(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_local_argmax_reduction:
'''
new = '''    def _greedy_sample_draft(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from b70_capture_hidden import capture
        capture("draft", hidden_states)
        if self.use_local_argmax_reduction:
'''
assert source.count(old) == 1, source.count(old)
speculator.write_text(source.replace(old, new, 1))

step = site / "v1/spec_decode/step3p5.py"
source = step.read_text()
old = '''    def _sample_draft_tokens_for_step(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
'''
new = old + '''        from b70_capture_hidden import capture
        capture("draft", hidden_states)
'''
assert source.count(old) == 1, source.count(old)
step.write_text(source.replace(old, new, 1))
