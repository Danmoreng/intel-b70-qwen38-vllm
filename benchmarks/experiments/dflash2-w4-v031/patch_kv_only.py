"""Separate opt-in packed KV projection; normal B8 query QKV is unchanged."""
import ast
import hashlib
from pathlib import Path
import sys

p=Path(sys.argv[1]);s=p.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='7892ea98bfd07de66c338752ce289736fef2e7d732635930e3c210eda26d92ed'
assert s.count('import io\n')==1;s=s.replace('import io\n','import io\nimport os\n')
old='''                qkv, _ = proj(normed_context_states)
                layer_kv.append(
                    qkv[:, q_size:].view(num_ctx, 2, num_kv_heads, head_dim)
                )'''
new='''                if os.environ.get("B70_DFLASH_KV_ONLY") == "1":
                    assert hasattr(proj.quant_method, "apply_context_kv")
                    kv = proj.quant_method.apply_context_kv(proj, normed_context_states)
                else:
                    qkv, _ = proj(normed_context_states)
                    kv = qkv[:, q_size:]
                layer_kv.append(kv.view(num_ctx, 2, num_kv_heads, head_dim))'''
assert s.count(old)==1;s=s.replace(old,new);ast.parse(s);p.write_text(s)
