# Current production configuration

[Back to README](../README.md)

The machine-readable authorities are [policy](../config/production_policy.json),
[image manifest](../config/production_image.json),
[RAM overlay](../config/production_ram_kv.json) and [.env.example](../.env.example).
The image release number and the policy schema version are separate identities.
The following table is checked against those manifests by
`python3 scripts/check-public-docs.py`.

<!-- BEGIN RELEASE IDENTITIES -->
| Identity | Pinned value |
| --- | --- |
| Image tag | `local/b70-qwen38-vllm:production-exl3-v7` |
| Image ID | `sha256:dc81825313d587ca22eaeeceb26808a8eed63cfb84b9dde22599223b8dab59d1` |
| Policy ID | `b70-qwen38-exl3-production-v5` |
| Policy SHA256 | `997d6b63458b312c038a1bd89034487d547e0368303c648aef6c44c69e9cae9f` |
| Checkpoint | `turboderp/Qwen3.8-27B-exl3` |
| Checkpoint revision | `113cf7ab958054860e43fb7f3063b1af19171095` |
| vLLM / Torch | 0.33.0 / 2.14.0+xpu |
| RAM layout / default budget | LBNHC / 12 GiB |
<!-- END RELEASE IDENTITIES -->

## Applied runtime

| Setting | Applied behavior |
|---|---|
| Weights / activations | EXL3 4.00-bpw checkpoint, mixed 4-bpw body / 6-bpw full target head; FP16 activations |
| Linear dispatch | Native SmallM through 128 rows; true MB4 for M≤4, original 24/40/48 choices; exact K4 decoder; INT8 large-matrix eager prefill |
| Prefill attention | Guarded exact-K oneDNN, query rows ≥64, active KV 4,096–262,144, query bucket 256, exact-K bucket 1; eager only |
| Verification | M04 at supported uniform q2–5 / C1–C4; direct output copy at C4; native fallback elsewhere; no duplicate KV updates |
| Partition cache | 64 exact-shape entries per inference thread/queue; completion-aware LRU; oneDNN's internal caches are separate |
| MTP | Three FP16 draft tokens; 65,536-row draft vocabulary, full 248,320-row target vocabulary |
| Admission | Sixteen slots, 4,096 max batched tokens, full input/output reservation |
| Scheduler watermark | Base policy 0.0; RAM overlay applies 0.05 |
| Graphs | FULL_DECODE_ONLY; captures at 1/2/4/8/12/16/24/32/40/48/56/64 |
| GPU cache | FP8 KV, prefix reuse, Mamba alignment; 1,792-token physical pages / 128-token prefix matching |
| COW | Complete dense-alias copy correction installed after actual KV initialization at every worker start |
| RAM cache | Inclusive host prefix cache; default 12 GiB, configurable integer budget 1–12 GiB; LBNHC layout |
| Transfers / host limit | 128 MiB staging batches; 23 GiB container memory limit, no swap allowance |
| GPU allocation / power | 0.965 memory fraction / 180 W card cap |
| Native toolchain | oneAPI 2026.1.1 / SYCL 9, oneDNN 3.13.4, Intel Runtime 26.35.39758.10, IGC 2.41.5 |
| Tools / reasoning | Automatic tool choice, qwen3_xml tool parser, qwen3 reasoning parser |
| Media | At most 32 images or 4 videos per prompt; image pixel cap 4,194,304 |

## API defaults and diagnostics

The default endpoint is `http://127.0.0.1:8081/v1`. The chat middleware applies
16,384 maximum completion tokens, an 8,192-token thinking budget, medium effort,
and retention of thinking history. A requested larger thinking budget is capped
at 8,192. `reasoning_effort: "none"` disables thinking unless explicitly
overridden by `chat_template_kwargs.enable_thinking`.

The launcher enables `--enable-per-request-metrics` and
`--enable-prompt-tokens-details`. Completed responses include server timing and
cache fields; streaming exposes them in the final usage event. The server's
prompt interval starts at scheduling and excludes queue time. Decode spans
first to last output token and uses N−1 tokens. Neither interval is an isolated
GPU-kernel measurement. An external agent gateway may retain these fields but
is not needed to use this API.

## Host options and integrity checks

Retain the frozen model, context, runtime and numerical settings in `.env`.
Host binding/port, render node and cache location are host-specific options;
`B70_KV_RAM_GIB` controls the allowed RAM budget. The launcher applies LBNHC
from the RAM manifest and rejects conflicting layout overrides.

Preflight verifies the image/policy, native libraries, Python runtime artifacts,
checkpoint files, middleware, serving arguments and power cap. Compiler caches
are separated by policy SHA and image ID. Developer/numerical endpoints are
absent from the production runtime. See [operations](OPERATIONS.md) for checks
and [installation](INSTALLATION.md) for the host-specific PCI power path.

[Current qualification and deployment evidence](../benchmarks/results/mtp-cache-1792-20261112/README.md).
