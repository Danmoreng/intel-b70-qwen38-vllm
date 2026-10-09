"""Opt-in group-four experiment, guarded to this exact hybrid geometry."""
import ast
import hashlib
from pathlib import Path
import sys

p=Path(sys.argv[1]);s=p.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='1b20c836aece4982dd87f7e06e14f5ae07a1fc2636101074cfad11584b1a0dcf'
needle='    grouped_layers = []\n    for layers in layer_buckets:\n'
replacement='''    if os.environ.get("B70_DFLASH_GROUP4") == "1":
        assert sorted(len(bucket) for bucket in layer_buckets) == [5, 16, 48]
        bucket_types = {len(bucket): type(specs[0]).__name__
                        for bucket, specs in zip(layer_buckets, spec_buckets)}
        assert bucket_types == {48: "MambaSpec", 16: "FullAttentionSpec",
                                5: "SlidingWindowSpec"}, bucket_types
        draft_names = next(bucket for bucket in layer_buckets if len(bucket)==5)
        assert all(any(f"layers.{i}." in name for i in range(64, 69))
                   for name in draft_names), draft_names
        group_size = 4
        logger.info("B70 opt-in DFlash group4: 12 Mamba, 4 full, 2 draft SWA groups; markers unchanged")
    grouped_layers = []
    for layers in layer_buckets:
'''
assert s.count(needle)==1;s=s.replace(needle,replacement);ast.parse(s);p.write_text(s)
