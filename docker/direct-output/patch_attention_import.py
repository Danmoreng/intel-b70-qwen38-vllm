from pathlib import Path

path=Path("/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py")
source=path.read_text()
old="from b70_attention import flash_attn_varlen_func"
new="from b70_ops.q128_attention import flash_attn_varlen_func"
assert source.count(old)==1
source=source.replace(old,new)
compile(source,str(path),"exec")
path.write_text(source)
