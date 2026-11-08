#!/usr/bin/env python3
"""Port the qualified EXL3 adapters to the SHA-pinned official vLLM 0.33 wheel.

Run on a fresh extraction of engine/exl3xpu/source.tar.gz, never production.
Native EXL3/M04 sources and math are unchanged; this only ports Python adapters.
"""
import argparse,ast,hashlib,json,shutil,zipfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',required=True,type=Path)
p.add_argument('--wheel',required=True,type=Path)
p.add_argument('--stock',required=True,type=Path)
p.add_argument('--policy',required=True,type=Path)
a=p.parse_args()
assert hashlib.sha256(a.wheel.read_bytes()).hexdigest()=='0e3735aa6b9ac132a32b54a948bb0de079e1b783ec5b04b184f4bdaf873eb995'
a.stock.mkdir(parents=True,exist_ok=False)
with zipfile.ZipFile(a.wheel) as z:z.extractall(a.stock)
v=a.stock/'vllm';s=a.source
BASE_SOURCE_SHA256 = {'exl3xpu/target_runtime.py': '55e0ec104708e636c36396fa57e474f9e477b2427d6f4c9414f8818c5f065e5d', 'exl3xpu/attention_dispatch.py': '7d6c3dea198388ab7d8be7dc73f5d135026125f5988653de0fb45d385fca0a41', 'exl3xpu/gdn_target_patch.py': '3b50c7076f5b91b19ca35d8ec43c7ccd482420ac5ddc01cb77bba2a3d115944b', 'exl3xpu/vllm_plugin.py': '5d47acc849c678092acc5231527028f842239db42ab55ab7ecfb5e6f89d6311c', 'models/qwen3.8-27b-exl3-4.00bpw/migration-target-optimized.yaml': 'c7bd0941664d02fa4ec4bb118efdecbf9c63801cb9e4cdfaeb366efbe8c8800d', 'tests/test_target_gdn.py': '031415aebb535e3b8907843d80b71a44dae3bf27125e65ba6ec5d9f599910e02'}
for name,digest in BASE_SOURCE_SHA256.items():
 assert hashlib.sha256((s/name).read_bytes()).hexdigest()==digest, 'Unreviewed EXL3 input: '+name
# EXL3 checkpoint-native MTP remains excluded from GPTQ-only dynamic overrides.
f=v/'model_executor/models/qwen3_5_mtp.py';text=f.read_text()
def once(text,old,new):
 assert text.count(old)==1,old
 return text.replace(old,new,1)
text=once(text,'import torch\n','import torch\nfrom .b70_mtp_policy import resolve_mtp_policy\n# B70_MTP_FORMAT_GUARD_V033\n')
text=once(text,'        quant_config = vllm_config.quant_config\n','        quant_config = vllm_config.quant_config\n        self._b70_mtp_policy = resolve_mtp_policy(quant_config, model_config.hf_config)\n        logger.info("B70 MTP quantization path: %s", self._b70_mtp_policy.description)\n')
text=once(text,'if quant_config and quant_config.get_name() not in ("modelopt_fp4",):','if self._b70_mtp_policy.is_gptq:')
compile(text,str(f),'exec');f.write_text(text)
shutil.copyfile(a.policy,f.parent/'b70_mtp_policy.py')
# Write only the pre-reviewed module boundary hashes from the pinned wheel.
# The diff review covers removed cascade metadata, unchanged native KV writes,
# fixed query partitions, exact prefill lengths and native decode lengths.
report={}
for name in ('target_runtime.py','attention_dispatch.py'):
 f=s/'exl3xpu'/name;text=f.read_text();tree=ast.parse(text)
 n=next(n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SOURCE_HASHES')
 old=ast.literal_eval(n.value)
 names = list(old) + (['model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'] if name == 'target_runtime.py' else [])
 new={k:hashlib.sha256((v/k).read_bytes()).hexdigest() for k in names}
 lines=text.splitlines(True);lines[n.lineno-1:n.end_lineno]=['SOURCE_HASHES = '+repr(new)+'\n'];text=''.join(lines)
 text=text.replace('0.31','0.33')
 if name=='attention_dispatch.py':
  text=once(text,'not md.use_cascade','not getattr(md, \'use_cascade\', False)')
 else:
  text=once(text,'def install():\n','def install():\n    if vllm.__version__ != \'0.33.0\':\n        raise RuntimeError(\'Unsupported EXL3 runtime version\')\n')
 f.write_text(text);report[name]={'old':old,'new':new}
f=s/'exl3xpu/gdn_target_patch.py';text=f.read_text().replace('reviewed 0.31','reviewed 0.33')
text=once(text,"'block_table_tensor': 3","'block_table_tensor': 2");f.write_text(text)
f=s/'exl3xpu/vllm_plugin.py';text=f.read_text();text=once(text,"== 'vllm031'","== 'vllm033'");f.write_text(text)
f=s/'models/qwen3.8-27b-exl3-4.00bpw/migration-target-optimized.yaml';text=f.read_text();text=once(text,'EXL3_TARGET_RUNTIME: vllm031','EXL3_TARGET_RUNTIME: vllm033');f.write_text(text)
# Upstream added checkpoint planning to the native builder. Stub that helper in
# the existing row-semantic test only; compare the actual new build bodies.
f=s/'tests/test_target_gdn.py';text=f.read_text().replace('0.31','0.33').replace('seven row sites','six remaining row sites')
text=once(text,"b.kv_cache_spec = None", "b.kv_cache_spec = None\n        b.gdn_prefill_backend = 'sycl'\n        b.checkpoint_builder = SimpleNamespace(build=lambda *args: None)\n        b.plan_flashinfer_checkpoints = False")
text=once(text,'([4], [3]), ([1], [-1]),','([4], [3]), ([1], [0]), ([1]*4, [0]*4), ([1], [-1]),')
text=once(text,"with patch.object(gdn_attn, 'mamba_get_block_table_tensor'","with patch('vllm.utils.torch_utils.PIN_MEMORY', device != 'cpu'), patch.object(gdn_attn, 'mamba_get_block_table_tensor'")
f.write_text(text)
(a.source.parent/'port-boundaries.json').write_text(json.dumps(report,indent=2)+'\n')
print('Ported guarded boundaries and retained six GDN mask sites. No native math changes.')
