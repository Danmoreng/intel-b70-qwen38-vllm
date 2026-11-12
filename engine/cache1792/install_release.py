"""Package the tested COW wrapper with automatic worker-start activation."""
import hashlib
from pathlib import Path
import yaml

ROOT=Path('/opt/venv/lib/python3.12/site-packages/vllm/v1/worker')
runner=ROOT/'gpu/model_runner.py'
assert hashlib.sha256(runner.read_bytes()).hexdigest()=='9145e2601f9e93db5e6c7e648c6741cff55f2a49f046800f79311e4977550121'
helper=ROOT/'b70_cow_copy_fix.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='b30646e39778904cb3f5ce2664d37defe7a743b8a4e6715d5b883f1ec54d5f43'
anchor='        self.model_state.initialize_kv_cache(self.kv_cache_config, self.block_tables)\n'
text=runner.read_text();assert text.count(anchor)==1
activation='''        # Same covering-alias wrapper used by the1792/128 qualification.
        # Profiling allocations must not install a second global wrapper.
        if not is_profiling:
            from vllm.v1.worker.b70_cow_copy_fix import install as install_complete_cow
            import json as _b70_json
            _b70_cow_receipt = install_complete_cow(self)
            print("B70_COMPLETE_COW_READY " + _b70_json.dumps(_b70_cow_receipt), flush=True)
'''
runner.write_text(text.replace(anchor,anchor+activation))
# Attention metadata code is unchanged. Retain its strict source guard, pinned
# to precisely the new runner with the startup-only activation above.
dispatch=Path('/opt/exl3xpu/exl3xpu/attention_dispatch.py')
assert hashlib.sha256(dispatch.read_bytes()).hexdigest()=='bd6a715c941f363ae6adc812bc6edf859589f788c7ef50a2ac2629d997f4f8dc'
old='9145e2601f9e93db5e6c7e648c6741cff55f2a49f046800f79311e4977550121'
guard=dispatch.read_text();assert guard.count(old)==1
dispatch.write_text(guard.replace(old,hashlib.sha256(runner.read_bytes()).hexdigest()))
profile=Path('/opt/exl3xpu/models/qwen3.8-27b-exl3-4.00bpw/migration-target-optimized.yaml')
assert hashlib.sha256(profile.read_bytes()).hexdigest()=='4fa3dd0304db3e3af016e1c90d392542143e3c3dc3fd56b77da50e6aae66ef08'
cfg=yaml.safe_load(profile.read_text())
cfg['vllm'].update(block_size=1792,prefix_match_unit=128,long_prefill_token_threshold=0)
profile.write_text(yaml.safe_dump(cfg,sort_keys=False))
