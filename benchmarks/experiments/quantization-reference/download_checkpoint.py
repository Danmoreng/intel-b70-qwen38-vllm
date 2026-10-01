"""Download a pinned GPTQ snapshot and verify files and panel tokenization."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--reference-tokenizer", required=True)
    args = p.parse_args()
    root = Path(args.root)
    assert not (root / f"{args.label}-checkpoint.json").exists()
    started = time.monotonic()
    api = HfApi().model_info(args.repo, revision=args.revision, files_metadata=True)
    assert api.sha == args.revision
    snapshot = Path(snapshot_download(args.repo, revision=args.revision, max_workers=2))
    checked = []
    for item in api.siblings:
        target = snapshot / item.rfilename
        assert target.stat().st_size == item.size, item.rfilename
        sha = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
        if item.lfs:
            assert sha == item.lfs.sha256, item.rfilename
        checked.append({"name": item.rfilename, "bytes": item.size,
                        "sha256": sha, "lfs_sha256_verified": bool(item.lfs)})
        print(json.dumps(checked[-1]), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    original = AutoTokenizer.from_pretrained(args.reference_tokenizer, local_files_only=True)
    assert tokenizer.get_vocab() == original.get_vocab(), "Token ID vocabularies differ"
    panel = json.loads((root / "panel.json").read_text())
    wiki = "\n".join(pq.read_table(root / "sources/wikitext-test.parquet")["text"].to_pylist())
    source_ids = {}
    for source in panel["sources"]:
        path = root / "sources" / source["name"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
        text = wiki if source["name"] == "wikitext-test.parquet" else path.read_text()
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert ids == original.encode(text, add_special_tokens=False), source["name"]
        source_ids[source["name"]] = ids
    for window in panel["windows"]:
        name = "wikitext-test.parquet" if window["domain"] == "prose" else window["name"]
        offset = window["source_start_token"]
        assert source_ids[name][offset:offset + 1024] == window["ids"]
    metadata = {"repo_id": args.repo, "revision": args.revision,
                "snapshot": str(snapshot), "files": checked,
                "total_bytes": sum(x["bytes"] for x in checked),
                "panel_sha256": hashlib.sha256((root / "panel.json").read_bytes()).hexdigest(),
                "tokenizer_vocabulary_equal": True, "tokenizer_source_encodings_equal": True,
                "panel_token_ids_verified": True, "seconds": time.monotonic() - started}
    (root / f"{args.label}-checkpoint.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
