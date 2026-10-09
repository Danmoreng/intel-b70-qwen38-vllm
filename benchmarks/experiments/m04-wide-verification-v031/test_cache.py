#!/usr/bin/env python3
"""CPU regression: speculative lengths must isolate compiled graph caches."""
import hashlib
import json
from pathlib import Path
from vllm.config.speculative import SpeculativeConfig

rows = []
for method in ('dflash', 'mtp'):
    hashes = []
    for depth in (3, 4, 7, 8):
        # Exercise the pure hash function without constructing a model or GPU.
        config = object.__new__(SpeculativeConfig)
        config.method = method
        config.num_speculative_tokens = depth
        config.draft_model_config = None
        digest = config.compute_hash()
        assert digest == config.compute_hash()
        hashes.append(digest)
        rows.append(dict(method=method, depth=depth, hash=digest))
    assert len(set(hashes)) == 4
source = Path(__import__('vllm.config.speculative', fromlist=['x']).__file__)
print(json.dumps(dict(status='PASS_SPECULATIVE_DEPTH_CACHE_ISOLATION',
                     source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     variants=rows), indent=2))
