"""Make a portable reference file: no original weights needed for later KL."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    panel_bytes = (root / "panel.json").read_bytes()
    panel = json.loads(panel_bytes)
    summary = json.loads((root / "bf16/summary.json").read_text())
    assert summary["panel_sha256"] == hashlib.sha256(panel_bytes).hexdigest()
    assert len(summary["windows"]) == len(panel["windows"])
    checkpoint = json.loads((root / "original-checkpoint.json").read_text())
    assert checkpoint["all_lfs_sha256_verified"]
    logprobs = np.stack([np.load(root / "bf16" / f"window-{i:03d}-logprobs.npy")
                        for i in range(len(panel["windows"]))])
    nll = np.stack([np.load(root / "bf16" / f"window-{i:03d}-nll.npy")
                   for i in range(len(panel["windows"]))])
    metadata = {"schema_version": 1, "panel": panel, "reference_summary": summary,
                "checkpoint": checkpoint, "logprobs_shape_axes": ["window", "sampled_position", "vocab_id"],
                "alignment": "logprobs[window,k] predicts input_ids[window,positions[window,k]+1]"}
    target = root / "reference-bf16.npz"
    np.savez_compressed(target, logprobs=logprobs, nll=nll,
        input_ids=np.asarray([w["ids"] for w in panel["windows"]], dtype=np.int32),
        positions=np.asarray([w["kl_positions"] for w in panel["windows"]], dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata)))
    with np.load(target, allow_pickle=False) as data:
        assert np.array_equal(logprobs, data["logprobs"])
        assert np.array_equal(nll, data["nll"])
    manifest = {"file": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "bytes": target.stat().st_size, "panel_sha256": summary["panel_sha256"],
                "logprobs_shape": list(logprobs.shape), "roundtrip_exact": True}
    (root / "reference-bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
