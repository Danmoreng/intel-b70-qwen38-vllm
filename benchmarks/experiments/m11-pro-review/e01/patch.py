"""Opt-in XPU dispatch on the pinned production source; CUDA is unchanged."""
from pathlib import Path
import difflib

path = Path('/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py')
old = path.read_text()
assert 'B70_FUSED_QK_ROPE_GATE' not in old
text = old.replace('import torch\n', 'import os\n\nimport torch\n', 1)
needle = '        self.use_fused_qk_norm_rope_gate = (\n'
assert text.count(needle) == 1
text = text.replace(needle, '''        self._b70_xpu_qk_fusion = (
            current_platform.is_xpu()
            and os.environ.get("B70_FUSED_QK_ROPE_GATE", "0") == "1"
            and tp_size == 1
            and self.num_heads == 24
            and self.num_kv_heads == 4
            and self.head_dim == 256
            and self.rotary_emb.rotary_dim == 64
            and self.rotary_emb.dtype == torch.float16
            and (mrope_section is None or (
                supports_mrope and list(mrope_section) == [11, 11, 10]
            ))
        )
''' + needle)
needle = '            and current_platform.is_cuda()\n'
assert text.count(needle) == 1
text = text.replace(needle, '            and (current_platform.is_cuda() or self._b70_xpu_qk_fusion)\n')
needle = '    def _project_qkv_gate(\n'
text = text.replace(needle, '''    def _b70_qk_metadata_ok(self, qkv, positions):
        # Metadata only: no device reads, implicit copies or launch-error fallback.
        cache = self.rotary_emb.cos_sin_cache
        q_weight, k_weight = self.q_norm.weight, self.k_norm.weight
        return (
            qkv.device.type == "xpu" and qkv.dtype == torch.float16
            and qkv.ndim == 2 and qkv.shape[0] > 0
            and qkv.shape[1] == 14336 and qkv.stride(-1) == 1
            and qkv.stride(0) >= 14336
            and positions.device == qkv.device
            and positions.dtype in (torch.int32, torch.int64)
            and positions.ndim in (1, 2)
            and positions.shape[-1] == qkv.shape[0]
            and positions.stride(-1) > 0
            and (positions.ndim == 1 or (
                positions.shape[0] == 3 and positions.stride(0) > 0
                and getattr(self.rotary_emb, "mrope_interleaved", False)
                and list(getattr(self.rotary_emb, "mrope_section", []) or []) == [11, 11, 10]
            ))
            and cache.device == qkv.device and cache.dtype == qkv.dtype
            and cache.ndim == 2 and cache.shape[1] == 64
            and cache.stride(-1) == 1 and cache.stride(0) >= 64
            and q_weight.device == qkv.device and k_weight.device == qkv.device
            and q_weight.dtype == qkv.dtype and k_weight.dtype == qkv.dtype
            and q_weight.shape == (256,) and k_weight.shape == (256,)
            and q_weight.stride(0) == 1 and k_weight.stride(0) == 1
        )

''' + needle)
text = text.replace('        if self.use_fused_qk_norm_rope_gate:\n', '''        if self.use_fused_qk_norm_rope_gate and (
            not self._b70_xpu_qk_fusion or self._b70_qk_metadata_ok(qkv, positions)
        ):
''', 1)
compile(text, str(path), 'exec')
path.write_text(text)
Path('/opt/b70-e01.patch').write_text(''.join(difflib.unified_diff(
    old.splitlines(True), text.splitlines(True), fromfile='a/qwen3_next.py', tofile='b/qwen3_next.py')))
print('Applied E01 opt-in XPU patch.')
