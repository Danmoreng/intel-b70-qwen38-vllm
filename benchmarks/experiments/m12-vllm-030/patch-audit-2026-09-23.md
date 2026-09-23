# vLLM 0.30 patch audit — 2026-09-23

Scope: the production `docker/Dockerfile`, the unmodified official XPU
`v0.30.0` image (`sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d`),
the M12 candidate image (`sha256:05e0981ea0a37ed82b309dbba3157f57c9a8144aadbf2bcb8746bfed70d95016`),
and the pinned model revision `a47b0c6f0d756bc394c4cc629d5b0ded1acc7001`.
The two image filesystems were compared directly at the modified source paths.
Upstream `main` was checked at `15ed1262e70873a65582d82f7973784a0f11e5a2`;
the `v0.30.1rc0` source at `153242a314153637999eb6ebe8dd830e63433bb6`.

## Release check

The latest [published stable release](https://github.com/vllm-project/vllm/releases)
is `v0.30.0` (2026-09-22). There is a `v0.30.1rc0` source tag, but no
`v0.30.1` stable release in the official releases list as of this audit.
The release candidate was not benchmarked or used for the production image.

## Active production patch inventory

| Component | Status in unmodified v0.30.0 | Consequence for this model |
| --- | --- | --- |
| Cookbook `patch_mtp_nightly.py` (`B70_MTP_BF16_DRAFT`) | The upstream [Qwen3.5 MTP code](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/qwen3_5_mtp.py#L106-L121) already disables MTP quantization when `quantization_config.dynamic` contains a `-:` key with `mtp`. | **Functionally redundant for the pinned checkpoint**: its config contains `"-:.*mtp.*"`. The custom environment gate is an alternate route to the same `vllm_config.quant_config = None`. Remove only in a separately validated image; do not assume the same for other checkpoints. |
| Cookbook `patch_mtp_boundary.py` | The unmodified [GDN attention path](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/attention/backends/gdn_attn.py#L293-L321) still uses the partial speculative group path that this patch changes. | Retain pending boundary-specific qualification. The short 4K/1K A/B did not reach the max-model-length boundary. |
| Cookbook `patch_fix_accepted_sync.py` | The V1 runner still has the original accepted-token copy/reorder code; [PR #53919](https://github.com/vllm-project/vllm/pull/53919) remains open. | The 0.30 test server logged **V2 Model Runner**, so this V1-only patch was inactive on the measured path. It protects a possible V1 fallback; removing it needs an explicit V2-only support decision. |
| Cookbook `patch_fix_backward_copy.py` | The unmodified [Mamba copy code](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/mamba_utils.py) lacks the three local backward-copy guards. The same guards are absent in the checked `main` and `v0.30.1rc0` files. | Retain until the upstream behavior and long-context/prefix-cache cases are qualified. |
| Cookbook `patch_fix_eagle_drop.py` | The unmodified [Mamba manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/single_type_kv_cache_manager.py) still ignores `drop_eagle_block` in its fine and coarse search branches; [PR #48375](https://github.com/vllm-project/vllm/pull/48375) remains open. | Retain. Our local patch covers both branches. The short A/B had zero prefix-cache hits and did not test this behavior. |
| Local `patch_draft_lmhead_int4.py` | The draft LM-head INT4 helper and its `compute_logits` routing are absent from the unmodified Qwen3.5 MTP code. | Retain for the current MTP configuration. The XPU INT4 GEMM primitive exists upstream, but quantizing/routing the draft LM head is local. |
| Local `patch_draft_mtp_int4.py` | The draft MTP layer INT4 helper and its `load_weights` routing are absent from the unmodified Qwen3.5 MTP code. | Retain for the current MTP configuration. |
| Q128/KV32 and M04 `.so` files plus `b70_attention.py` | These are local extensions; the Dockerfile changes `_xpu_ops.py` to import the adapter. | Retain for the current MTP-enabled stack. The M12 MTP benchmark confirmed dispatch; the no-MTP benchmark did not dispatch either kernel in either version. |

`patch_reduced_draft_vocab.py` is present in the `docker/patches/` build
context, but the production Dockerfile never executes it. It is **not** an
active production patch.

The direct filesystem comparison showed that every one of the seven Python
patch scripts applied a source change to the v0.30.0 image. This is different
from runtime necessity: the BF16 draft gate duplicates an upstream behavior
for this model, and the accepted-sync patch only changes the unused V1 runner
in the measured configuration. The checked `main` and `v0.30.1rc0` source
files do not contain the local boundary, backward-copy, or EAGLE-drop changes.

## Upgrade implication

The published v0.30.0 candidate with all production patches ran successfully.
It was **11.03% slower** than production v0.29 on the MTP-enabled 4K/1K C4
screen. Disabling MTP removes that version gap, but also reduces aggregate C4
decode from 231.7 tok/s (v0.29 with MTP) to about 112 tok/s on either version
for the same prompts. The measurements point to an interaction in the MTP
path; they do not identify which component causes it. The current production
image remains the sensible choice until the MTP regression and the relevant
correctness/long-context behavior are qualified. No production image or
service was changed by this audit.
