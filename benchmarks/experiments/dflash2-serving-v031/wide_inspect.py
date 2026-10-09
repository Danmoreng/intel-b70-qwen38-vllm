"""Attest the widened M04 artifact before timing actual DFlash serving."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import torch
from dflash_inspect import WorkerExtension as Base

class WorkerExtension(Base):
    def inspect_dflash_setup(self):
        result = super().inspect_dflash_setup()
        from exl3xpu import shared_kv_verify as verify
        source = inspect.getsource(verify.eligible)
        assert 'not 2 <= rows <= 9' in source
        assert 'adaptive_splits' in inspect.getsource(verify.run)
        library = Path(os.environ['EXL3_M04_LIBRARY'])
        digest = hashlib.sha256(library.read_bytes()).hexdigest()
        assert digest == os.environ['EXL3_M04_LIBRARY_SHA256']
        schema = str(torch._C._dispatch_find_schema_or_throw('b70_exl3_attention::shared_kv_verify_out', '').schema())
        assert 'bool adaptive_splits=False' in schema
        result.update(m04_max_query_rows=9, verification_query_rows=result['num_speculative_tokens'] + 1,
                      m04_library_sha256=digest, m04_wrapper_sha256=hashlib.sha256(Path(verify.__file__).read_bytes()).hexdigest(),
                      m04_operator_schema=schema, gpu_length_adaptive_splits=True)
        Path('/results/draft-identity.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
