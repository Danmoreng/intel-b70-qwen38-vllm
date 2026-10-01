"""Full-vocabulary forward/reverse KL, JS, PPL and paired window uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def normalized_logprobs(x):
    x = x.astype(np.float64)
    assert np.isfinite(x).all()
    z = np.logaddexp.reduce(x, axis=-1, keepdims=True)
    assert np.max(np.abs(z)) < 1e-4, "Saved distribution is not normalized"
    return x - z


def interval(values):
    rng = np.random.default_rng(20261001)
    values = np.asarray(values)
    draws = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return np.quantile(draws, [.025, .975]).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--arms", nargs="+", default=["fp16", "gptq", "exl3"])
    p.add_argument("--output-name", default="comparison.json")
    p.add_argument("--tokenizer", help="Frozen tokenizer JSON; report probability outside its IDs without masking any arm")
    args = p.parse_args()
    root = Path(args.root)
    panel_bytes = (root / "panel.json").read_bytes()
    panel = json.loads(panel_bytes)
    panel_sha = hashlib.sha256(panel_bytes).hexdigest()
    reference = json.loads((root / "bf16/summary.json").read_text())
    assert reference["panel_sha256"] == panel_sha
    n = len(reference["windows"])
    invalid_ids = None
    vocabulary = {"logit_rows": 248320, "masking": "No arm is additionally masked by this comparison"}
    if args.tokenizer:
        tokenizer_bytes = Path(args.tokenizer).read_bytes()
        tokenizer = json.loads(tokenizer_bytes)
        ids = set(tokenizer["model"]["vocab"].values()) | {r["id"] for r in tokenizer.get("added_tokens", [])}
        assert min(ids) >= 0 and max(ids) < 248320
        invalid_ids = np.array(sorted(set(range(248320)) - ids), dtype=np.int64)
        vocabulary.update(tokenizer_sha256=hashlib.sha256(tokenizer_bytes).hexdigest(),
                          tokenizer_ids=len(ids), max_tokenizer_id=max(ids),
                          logit_rows_outside_tokenizer=len(invalid_ids))
    result = {"panel_sha256": panel_sha, "reference": reference,
              "kl_direction": "KL(original || checkpoint), nats; full 248320-token vocabulary",
              "confidence_interval": "95% paired bootstrap over windows; exploratory small panel",
              "precision_note": "Log-probabilities saved as FP32, renormalized in FP64 for comparison",
              "vocabulary": vocabulary, "arms": {}}
    for arm in args.arms:
        meta = json.loads((root / arm / "summary.json").read_text())
        assert meta["panel_sha256"] == panel_sha
        assert len(meta["windows"]) == n
        window_rows = []
        for i, w in enumerate(panel["windows"][:n]):
            original_nll = np.load(root / "bf16" / f"window-{i:03d}-nll.npy").astype(np.float64)
            other_nll = np.load(root / arm / f"window-{i:03d}-nll.npy").astype(np.float64)
            assert original_nll.shape == other_nll.shape == (1023,)
            ref = normalized_logprobs(np.load(root / "bf16" / f"window-{i:03d}-logprobs.npy"))
            other = normalized_logprobs(np.load(root / arm / f"window-{i:03d}-logprobs.npy"))
            assert ref.shape == other.shape == (32, 248320)
            prob, quant = np.exp(ref), np.exp(other)
            kl = (prob * (ref - other)).sum(axis=-1)
            reverse = (quant * (other - ref)).sum(axis=-1)
            middle = np.logaddexp(ref, other) - np.log(2)
            js = .5 * ((prob * (ref - middle)).sum(axis=-1) + (quant * (other - middle)).sum(axis=-1))
            assert np.min(kl) >= -1e-10 and np.min(reverse) >= -1e-10
            row = {"window": i, "name": w["name"], "domain": w["domain"],
                   "original_nll_mean": float(original_nll.mean()),
                   "nll_mean": float(other_nll.mean()),
                   "delta_nll": float((other_nll - original_nll).mean()),
                   "kl_original_to_checkpoint": kl.tolist(),
                   "kl_checkpoint_to_original": reverse.tolist(),
                   "js_divergence": js.tolist(),
                   "top1_agreement": float(np.mean(ref.argmax(-1) == other.argmax(-1)))}
            if invalid_ids is not None:
                row["original_probability_outside_tokenizer_mean"] = float(prob[:, invalid_ids].sum(-1).mean())
                row["checkpoint_probability_outside_tokenizer_mean"] = float(quant[:, invalid_ids].sum(-1).mean())
            window_rows.append(row)

        def aggregate(rows):
            kl = np.concatenate([r["kl_original_to_checkpoint"] for r in rows])
            rev = np.concatenate([r["kl_checkpoint_to_original"] for r in rows])
            nll = np.mean([r["nll_mean"] for r in rows])
            delta = np.mean([r["delta_nll"] for r in rows])
            aggregated = {"windows": len(rows), "ppl_positions": len(rows) * 1023,
                    "kl_positions": len(kl), "nll_mean": float(nll),
                    "perplexity": float(np.exp(nll)), "delta_nll": float(delta),
                    "perplexity_ratio_to_original": float(np.exp(delta)),
                    "kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)),
                    "kl_p95": float(np.quantile(kl, .95)), "kl_p99": float(np.quantile(kl, .99)),
                    "reverse_kl_mean": float(rev.mean()),
                    "js_mean": float(np.mean([r["js_divergence"] for r in rows])),
                    "top1_agreement": float(np.mean([r["top1_agreement"] for r in rows])),
                    "delta_nll_ci95": interval([r["delta_nll"] for r in rows]),
                    "kl_mean_ci95": interval([np.mean(r["kl_original_to_checkpoint"]) for r in rows])}
            if invalid_ids is not None:
                for field in ("original_probability_outside_tokenizer_mean", "checkpoint_probability_outside_tokenizer_mean"):
                    aggregated[field] = float(np.mean([r[field] for r in rows]))
            return aggregated

        result["arms"][arm] = {"metadata": meta, "overall": aggregate(window_rows),
            "domains": {domain: aggregate([r for r in window_rows if r["domain"] == domain])
                        for domain in sorted({r["domain"] for r in window_rows})},
            "windows": window_rows}
        print(arm, json.dumps(result["arms"][arm]["overall"]), flush=True)
    if "gptq" in result["arms"] and "exl3" in result["arms"]:
        a, b = result["arms"]["gptq"]["windows"], result["arms"]["exl3"]["windows"]
        differences = [x["nll_mean"] - y["nll_mean"] for x, y in zip(a, b)]
        result["gptq_minus_exl3"] = {"delta_nll": float(np.mean(differences)),
                                      "delta_nll_ci95": interval(differences)}
    if "exl3" in result["arms"]:
        old = result["arms"]["exl3"]["windows"]
        result["paired_new_minus_old_exl3"] = {}
        for label, arm in result["arms"].items():
            if not label.startswith("target-"):
                continue
            differences = [x["nll_mean"] - y["nll_mean"] for x, y in zip(arm["windows"], old)]
            result["paired_new_minus_old_exl3"][label] = {
                "delta_nll": float(np.mean(differences)),
                "perplexity_ratio": float(np.exp(np.mean(differences))),
                "delta_nll_ci95": interval(differences)}
    assert Path(args.output_name).name == args.output_name
    (root / args.output_name).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
