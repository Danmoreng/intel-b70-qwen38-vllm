"""Freeze 16 teacher-forced windows; never use model-generated continuations."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import urllib.request

import pyarrow.parquet as pq
from transformers import AutoTokenizer

ORIGINAL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
GPTQ_REVISION = "a47b0c6f0d756bc394c4cc629d5b0ded1acc7001"
WIKI_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--gptq", required=True)
    p.add_argument("--exl3", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    (out / "sources").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.original, local_files_only=True)
    others = [AutoTokenizer.from_pretrained(x, local_files_only=True) for x in (args.gptq, args.exl3)]
    assert all(t.get_vocab() == tok.get_vocab() for t in others), "Token ID vocabularies differ"
    sources = []

    def fetch(name, url):
        target = out / "sources" / name
        if not target.exists():
            with urllib.request.urlopen(url, timeout=90) as r:
                target.write_bytes(r.read())
        data = target.read_bytes()
        sources.append({"name": name, "url": url, "sha256": digest(data), "bytes": len(data)})
        return target

    wiki = fetch("wikitext-test.parquet", f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{WIKI_REVISION}/wikitext-2-raw-v1/test-00000-of-00001.parquet")
    texts = pq.read_table(wiki)["text"].to_pylist()
    wiki_text = "\n".join(texts)
    wiki_ids = tok.encode(wiki_text, add_special_tokens=False)
    assert all(t.encode(wiki_text, add_special_tokens=False) == wiki_ids for t in others)
    rng = random.Random(20261001)
    windows = []

    def add(name, domain, ids, start):
        ids = ids[start:start + 1024]
        assert len(ids) == 1024
        positions = sorted(rng.sample(range(128, 1023), 32))
        windows.append({"name": name, "domain": domain, "source_start_token": start,
                        "ids": ids, "kl_positions": positions,
                        "token_ids_sha256": digest(json.dumps(ids, separators=(",", ":")).encode())})

    for i in range(8):
        start = (len(wiki_ids) - 1024) * i // 7
        add(f"wikitext-test-{i:02d}", "prose", wiki_ids, start)
    code_sources = [
        (f"python-{name}.py", f"https://raw.githubusercontent.com/python/cpython/v3.12.10/Lib/{name}.py")
        for name in ("ast", "tokenize", "configparser", "pathlib")
    ] + [
        (f"vue-{name}.ts", f"https://raw.githubusercontent.com/vuejs/core/v3.5.13/packages/reactivity/src/{name}.ts")
        for name in ("baseHandlers", "reactive", "computed", "effect")
    ]
    for name, url in code_sources:
        text = fetch(name, url).read_text()
        ids = tok.encode(text, add_special_tokens=False)
        assert all(t.encode(text, add_special_tokens=False) == ids for t in others)
        assert len(ids) >= 1024, name
        start = rng.randrange(len(ids) - 1024 + 1)
        add(name, "code", ids, start)
    panel = {"schema_version": 1, "seed": 20261001, "window_length": 1024,
             "prediction_alignment": "logits at input index j predict ids[j+1]",
             "ppl_positions_per_window": 1023, "kl_positions_per_window": 32,
             "kl_sampling": "uniform without replacement over input indices 128..1022",
             "original_revision": ORIGINAL_REVISION, "gptq_revision": GPTQ_REVISION,
             "sources": sources, "windows": windows,
             "tokenizer_vocabulary_equal": True, "tokenizer_source_encodings_equal": True}
    data = (json.dumps(panel, indent=2) + "\n").encode()
    (out / "panel.json").write_bytes(data)
    print(json.dumps({"panel": str(out / "panel.json"), "sha256": digest(data),
                      "windows": len(windows), "predictions": len(windows) * 1023,
                      "full_vocab_kl_positions": len(windows) * 32}), flush=True)


if __name__ == "__main__":
    main()
