# EXL3 follow-up review and bounded release checklist

**Review date:** 2026-10-02  
**Input:** `B70_EXL3_Review_Followup_Final_2026-10-02.zip`  
**Decision:** Proceed to candidate serving benchmarks and release qualification. Promote only after those checks pass. No additional open-ended kernel optimization campaign is warranted by this review.

## 1. Release identities and scope

| Role | Identity |
|---|---|
| Currently qualified EXL3 v1 image | `sha256:c09015ce22180fbc90ef0f5070f4a7116c8d11be785067499477accf0216f21f` |
| Reviewed candidate runtime image | `sha256:0e711fea1f9231a25289d812fffbde51ed93cbe7bad16c34f7fde3edf3d91737` |
| Candidate tag | `local/b70-qwen38-vllm:exl3-review-cache-c4copy-v2` |
| Candidate native EXL3 library SHA256 | `e60e499aefa95b290924ed98545eb815aca9d0fb1f3ede39b37d1ad49f3b522d` |
| Unchanged M04 library SHA256 | `eaa18427db27d4fceeca8a17c8a3c6b019b7678f39bf98affc1736b9f2c2d631` |
| Candidate EXL3 source commit, as recorded | `c59d9442aba8610188837e37724600f1517d7335` |
| Owned repository commit, as recorded | `27c363727ba8acd5774247282af1bf3b63935cde` |

The image and commit identities above are recorded in the supplied archive; no live Docker daemon or remote repository was inspected. The candidate is not yet a qualified production release. If release labels/policy metadata produce a new image ID, record and verify the parent/child relationship and benchmark or smoke-test the final deployable identity as specified below.

## 2. Independent checks performed in this review

- All **1,345 manifest-listed files** match their recorded sizes and SHA256 digests.
- All **13 active source-guard dependencies** are included and match the installed-runtime capture. All **118 files covered by the capture's file-hash inventory** also match. The union of the candidate's two source-guard declarations matches that coverage.
- The archived candidate patch applies to the archived qualified source and reproduces **all included candidate files exactly**.
- All **72 included files** from the candidate build receipt's 74-source inventory match their build-time hashes. Two build Dockerfiles are omitted; see section 4.
- All six native build source/manifest inputs match the candidate native build receipt.
- **28 of 29 exact receipt references** in the compact assessment are available and hash-correct. The missing reference is a historical runtime-source coverage receipt; a newer independently valid capture is included instead.
- Local Python execution: **31 tests passed; three environment-dependent tests were skipped**. This comprises 15 owned-repository tests, five attention-metadata tests, ten checkpoint-inventory tests and the M04 copy test. The latter internally covers 12 batch/query combinations with and without a supplied output tensor.
- The standalone C++ bounded-cache policy test compiled with a host compiler and passed. This is not a SYCL/native extension build.
- The GDN AST transformation accounts for its seven declared sites in the supplied guarded runtime source.
- The recorded GPU evidence was checked for identities, numerical flags, shapes and internal consistency. No new B70 test, native SYCL build, image startup or deployment was performed here.

The first broad owned-test discovery needed the repository's `scripts` directory on `PYTHONPATH`; rerunning with that path succeeded. The initial import-error log is retained rather than silently removed.

Reproduction of local checks, from the extracted archive:

```bash
cd engine-code
PYTHONPATH="$PWD/scripts:$PWD" python -m unittest discover -s tests/unit -v
cd ../exl3-candidate
PYTHONPATH="$PWD" python -m unittest discover -s tests -p test_attention_metadata.py -v
PYTHONPATH="$PWD" python -m unittest discover -s tests -p test_checkpoint_inventory.py -v
PYTHONPATH="$PWD" python -m unittest discover -s tests -p test_m04_output_copy.py -v
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I csrc \
  tests/test_bounded_partition_cache.cpp -o /tmp/exl3-cache-policy-test
/tmp/exl3-cache-policy-test
```

## 3. Disposition of the previous findings

### F1 — Build/restore helpers: addressed

`engine-code/scripts/build-image.sh:4–8` no longer derives its build tag from the serving `.env`. It uses a separate `B70_BASE_BUILD_IMAGE` and checks it before download/build side effects. `release_integrity.py:12–37` validates current release receipts and protects known current/rollback tags.

`restore-production.py:16–36` verifies the service working directory, exact launcher and strict launcher preflight before starting the service. The historical hard-coded service/image restoration path is gone. All seven focused operational-helper tests pass locally.

### F2 — Application oneDNN partition cache: addressed, with an explicit finite scope

The candidate adds `csrc/bounded_partition_cache.h` and uses a bounded exact-shape cache. Default capacity is 64 entries per thread/queue context; `EXL3_SDPA_CACHE_CAPACITY` accepts 1–256. Hits do not call a GPU completion wait. A miss under capacity pressure first chooses a completed least-recently-used entry; it waits for the oldest entry only when every entry is busy.

`exl3-candidate/csrc/exl3_ops.sycl:403–433,435–509,568–572` records each partition's completion event and chains subsequent uses to it. This is consistent with oneDNN's documented SYCL interop API, which accepts event dependencies and returns a completion event. The production prefill guard excludes graph capture (`attention_dispatch.py:59–62`); this review does not generalize the cache's lifetime safety to arbitrary graph-captured oneDNN callers.

The recorded finite GPU soaks use 81 exact lengths over three epochs, at capacities 8 and 64. Both caps hold, hits/evictions occur, and the original sampled numerical checks pass. Epoch-two-to-three RSS growth is 978,944 and 958,464 bytes, respectively; recorded XPU allocated/reserved growth is zero. No real GPU pressure-wait branch occurred in those runs; the host policy test exercises that branch.

Important limitations: this is not a global cap across arbitrarily many threads/queues, does not bound oneDNN's internal caches, and is not a permanent no-OOM guarantee. The soak's numerical sampling checks the repeated hot 4096-key result, not every varied-length output. These limitations do not invalidate the cache implementation; observe the actual serving workload during qualification.

### F3 — Numerical diagnosis: sufficiently addressed for the agreed bounded follow-up

Eight recorded matched-state target-step snapshots include actual prefill, mixed batches, real C4 q4 verification, and the two requested early output positions. Selected follow-ups restore and byte-verify the entire 11,196,825,600-byte backing cache storage. Full 248,320-vocabulary target logits are compared, not merely top-k values.

All **256 local attention comparisons** satisfy the original thresholds. Native-repeat is bitidentical in the recorded pure C4 verification graphs. No sampled target-logit row changes its top-1 token across the compared routes. Mixed native-repeat itself varies; the additional capture locates a small difference at the first GDN b/a projection, before attention. That supports numerical sensitivity as an explanation within these captured cases, rather than proving identical behavior for every historical trajectory.

**Identity boundary:** these full-model matched-state diagnostics deliberately run on qualified v1 (`c09015…`), not the final candidate. They resolve the prior v1 diagnostic question. Candidate-specific cache/attention/copy gates are separate. Neither collection replaces a fresh end-to-end startup and serving check of the final candidate.

Sources: `review-evidence/matched-state-*/campaign.json`, successful `replay/steps/*-result.json`, `engine-code/benchmarks/experiments/exl3-review-20261002/state_replay.py:49–159,183–364`.

### F4 — Active runtime source closure: addressed

The previously missing V2 runner, attention utilities and hybrid Mamba state code are now present. The collector derives the guard union and rejects missing, conflicting or changed source dependencies. The included negative-test receipt records missing-source and changed-hash rejection.

The separate historical receipt mismatch in section 4 is a packaging/provenance issue, not a failed hash check of the included runtime source.

### P1 — M04 output copy: conservative implementation and correctly limited claims

`shared_kv_verify.py:110–120` uses the direct permuted-view copy only for C4. C1–C3 retain the original unpack/copy route, and the no-output fallback remains. Shape/contiguity/device checks remain in `eligible()`.

The final candidate has 16 recorded C1/C4 × q2–5 × short/long-KV cases passing old/new equality, native tolerance, fallback equality, output-storage preservation and changed-input graph replay. The earlier unrestricted-copy q4 repeat reports about **0.762%** paired mean micro-latency speedup for short-KV C4; its round-bootstrap interval is approximately **0.680–0.847%**. C1 was slightly worse in that longer repeat, motivating the restriction. Long-KV intervals include zero. The shorter final C4-only timing confirmation is noisy.

Do not publish these numbers as serving decode-throughput gains. Graph timing and the final serving workload still need measurement; unchanged C1 code also exhibits timing variation in the short microtest.

### P2 — Route selection: addressed

`attention_dispatch.py:92–131,204–216` makes one checked route decision for logging and execution. Mixed subgroups still receive their own checked decision; unsupported cases retain native fallback. The recorded attention gate has 31 cases, two mutable-graph cases and three future-token poison cases under unchanged tolerances.

## 4. Remaining small corrections — not kernel blockers

### A. Include the actual build recipes

`build-review-bundle.py:55,74` allows the exact filename `Dockerfile`, but not `Dockerfile.*`. Consequently `exl3-candidate/docker/Dockerfile.migration-target` (1,223 bytes) and `Dockerfile.migration-old` (482 bytes) are omitted even though the build receipt hashes them and the final build command uses the target recipe.

Accept `Dockerfile` and `Dockerfile.*` in both source-copy filters, include the files, and verify their existing build-receipt hashes. Do not rebuild the candidate merely to repair the ZIP. Until included, the archived build-input inventory is not fully reconstructible.

### B. Preserve the exact historical coverage receipt

The assessment references `runtime-sources/SOURCE_GUARD_COVERAGE.json` at SHA256 `51ad7a2b21a6d6c87fc08361ce83d26d5b5937018c3fa07eaa028aafd489b944`. The archive deliberately omits that original receipt as a duplicate. The included `qualified-runtime/SOURCE_GUARD_COVERAGE.json` instead hashes to `247722088ed64c3ba7b86393c3fb33c256805fec864a0269e3a583715a521af5` and independently passes its 13/13 guards and 118 file hashes.

Retain the exact original receipt as a small historical evidence file, or explicitly regenerate/link the assessment to the new capture with its own provenance. Do not silently substitute a different file under the old digest. No GPU work is needed.

### C. Avoid the historical all-in-one benchmark orchestration

`run-exl3-final-readme.py:85–119` still unconditionally runs both GPTQ and EXL3 Flappy campaigns. That is historical orchestration, not a newly discovered serving bug. Do not invoke it unchanged merely to refresh current EXL3 serving numbers.

Also, `run-exl3-final-performance.py` compares native versus optimized attention within an image. The release comparison now needed is **qualified EXL3 v1 versus the candidate, both with their intended optimized production profiles**. Reuse the lower-level serving runner or make a narrow serving-only controller; do not weaken release checks to make old orchestration accept new receipts.

## 5. Bounded benchmark and release checklist for Codex

### Phase A — Freeze and repair provenance

- [ ] Apply the two packaging-only fixes above and rerun manifest/source checks.
- [ ] Keep candidate runtime `0e711f…` and native library `e60e49…` fixed. Keep quantization, MTP3, draft vocabulary, attention formulas, split table, driver and compiler versions unchanged.
- [ ] Record the effective cache capacity (64) in the candidate qualification configuration/receipt. If making it explicit in the deployed profile changes bytes or image metadata, pin that final configuration before qualification and document the relationship.
- [ ] Preserve qualified EXL3 v1's image/tag, policy, image manifest, runtime-source snapshot and cache namespace as the immediate rollback release. Do not overwrite its immutable tag.
- [ ] Prepare a distinct candidate/release attestation. Update the native library hash, source identity, image identity and receipt paths; do not change the current production qualification prematurely or disable strict preflight checks.

### Phase B — Run serving qualification, not another research campaign

Use exclusive GPU access and a bounded controller with failure cleanup/restoration. Do not run benchmark and production workers concurrently on the card. Keep fresh output directories, frozen prompts, fixed request settings, the 180 W cap, explicit cache conditions and exact image identities.

- [ ] Fresh candidate startup: full 409/409 loader inventory including 8/8 MTP modules; expected attention routes; no source-guard failures; successful real requests. Verify tools/reasoning/media defaults with bounded smoke cases.
- [ ] Run the existing **20-scenario / 70-wave serving matrix once** on the final deployable candidate. Do not attach the historical Flappy/GPTQ campaign. Preserve the isolated 64K prefix resend so a supposedly cold prefix is not warmed by preceding scenarios.
- [ ] Use a small matched v1/candidate subset to interpret regressions: short C1, long C1, short/moderate C4, and mixed serving. Both arms must use their intended optimized profile, identical prompts/settings and clearly separated warmup. Repeat an apparent regression once under matched conditions rather than launching a sweep.
- [ ] Exercise mixed decode plus incoming long prefill and a bounded series of varied exact prompt lengths in the same worker. Record cache entries/capacity, hits/misses, evictions, pressure waits and compile time from the actual worker/queue, plus RSS, XPU allocated/reserved memory, TTFT and p95 client stream gaps. Cache statistics read from a separate process are not statistics for the serving worker. Diagnostic instrumentation is not throughput evidence.
- [ ] Recheck the advertised long-context boundary: 261,120 input + 1,024 output tokens, moderate C4, prefix reuse, request abort/recovery and worker restart. Include one bounded C16 pressure case if retaining that operating claim; known C16 preemption allowance is not a no-preemption promise.
- [ ] Run a compact existing-reference numerical/quality smoke on the final candidate, including actual long-prefill and graph-verification exercise where appropriate. Reuse existing reference material; do not regenerate BF16 or repeat the whole historical quantization campaign. Never widen numerical tolerances to approve a failure.

Do not retune kernels, reopen MTP/vocabulary comparisons, rerun the old adaptive coding campaign, or add a driver/toolchain change unless a concrete failed release check warrants it.

### Phase C — Apply explicit release criteria

Promote only if identities/guards and required tests pass, ordinary and declared non-pressure scenarios have no unexpected OOMs/preemptions, API behavior is intact, abort/restart succeeds, and no unresolved material serving regression remains.

A proposed bounded review trigger is a reproducible **greater-than-5%** decode/TTFT regression or **greater-than-10%** mixed p95 stream-gap regression against matched v1 measurements. These are suggested decision thresholds, not previously proven SLAs. A noisy single result is not a kernel failure; repeat the affected matched case once and report uncertainty. Do not demand a measurable throughput gain: bounded resource retention and operating safety can justify a release with equivalent serving performance.

Write a release decision containing exact candidate/deployment IDs, library/source hashes, request/config identities, measured results, failed or skipped checks, known limitations and rollback steps. Never relabel v1 throughput or matched-state diagnostic numbers as new candidate measurements.

### Phase D — Deploy only the qualified identity

- [ ] Switch policy/image/environment consistently through the strict launcher; verify final image/labels/library hashes and service working directory.
- [ ] If a metadata-only release child is used, verify identical runtime filesystem layers and documented environment changes. Prefer benchmarking that final child; at minimum perform final startup/API/restart checks on it and clearly attribute any inherited benchmark evidence.
- [ ] Confirm `/v1/models`, a real generated response, one streaming request and the expected API defaults after deployment. Service process activity alone is insufficient.
- [ ] Exercise recovery to the saved EXL3 v1 release, or retain a verified immediately usable rollback path, without overwriting v1's tag or evidence. Restore v1 on any failed candidate promotion.
- [ ] Update the README's current image and serving measurements from the new receipts. Keep historical v1/Flappy/GPTQ evidence in separate documents and date-label any inherited quality results.

## 6. Overall verdict

The previous technical follow-up is substantially and competently implemented. The cache change is bounded and asynchronous-lifetime-aware, the direct copy was restricted after unfavorable C1 evidence, the numerical diagnosis is materially stronger, and the active runtime-source gap is closed. No newly proven severe inference defect was found in the reviewed delta.

The remaining work is **provenance cleanup, current serving measurements, exact-image release qualification and conditional promotion**, not another open-ended kernel investigation.

## External API cross-check

The event-dependency interpretation was cross-checked against the primary oneDNN API documentation: `dnnl::graph::sycl_interop::execute` accepts an optional event-dependency vector and returns an output event. This corroborates the API usage, not a fresh execution test of the pinned library.

Source: https://uxlfoundation.github.io/oneDNN/namespace_dnnl_graph_sycl_interop.html
