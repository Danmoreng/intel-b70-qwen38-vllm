#!/usr/bin/env python3
"""Keep target inventory strict, require checkpoint MTP only for MTP serving."""
import ast
import hashlib
import sys
from pathlib import Path
path=Path(sys.argv[1]);source=path.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='e1ce7c8ce6ef6a6fb1de8bbd62b1bc504471e67e51d0a888ea8fabef5d95fd92'
old="    required={k for k in expected if not k.startswith('mtp.') or draft is not None}"
new="""    spec=getattr(runner.vllm_config,'speculative_config',None)
    method=str(getattr(spec,'method','disabled')).lower()
    mtp_required='mtp' in method
    required={k for k in expected if not k.startswith('mtp.') or mtp_required}"""
assert source.count(old)==1;source=source.replace(old,new)
old="'mtp_skipped_reason':None if draft is not None else 'speculation disabled; no draft model instantiated'"
new="'mtp_skipped_reason':None if mtp_required else f'MTP not configured (speculative method={method})'"
assert source.count(old)==1;source=source.replace(old,new);ast.parse(source);path.write_text(source)
print('AUDIT_PATCH_SHA256',hashlib.sha256(source.encode()).hexdigest())
