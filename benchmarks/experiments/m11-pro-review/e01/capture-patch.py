from pathlib import Path
p=Path('/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py')
s=p.read_text()
needle='        self._b70_xpu_qk_fusion = (\n'
assert s.count(needle)==1
s=s.replace(needle,'        self._b70_capture_prefix = prefix\n'+needle)
needle='        q, k, v, gate = self._project_qkv_gate(qkv, positions)\n'
assert s.count(needle)==1
s=s.replace(needle,'        from b70_e01_capture import capture\n        capture(self, qkv, positions)\n'+needle)
compile(s,str(p),'exec');p.write_text(s)
