# B70 oneDNN production release checklist

The candidate is the single policy in `config/production_policy.json`.
Historical Q128/W4A16 results remain offline controls. Do not publish or
deploy an image merely because one preceding stage passed.

## C0 — Freeze the candidate

- [x] Select W4A8 at at least 512 total matrix rows, oneDNN for eligible
  Q=256..6656 and exact active KV=16384..196608, and mixed routing.
- [x] Preserve GPTQ model revision, FP8 KV, MTP4, full draft vocabulary,
  existing target head, draft quantization, 180 W, 200704 context, 6656 batch,
  C4 admission and 0.93 memory fraction.
- [x] Record one policy file and its SHA-256 in `config/production_policy.sha256`.
- [x] Freeze quality and serving acceptance rules below before new evaluation.

## C1 — Integrate and package

- [x] Install the mixed route in the canonical adapter and one release image.
- [x] Enforce one policy at launcher and adapter entry points; incompatible
  flags, image, model revision and library hashes fail startup.
- [x] Namespace AOT caches by image ID and policy hash; record actual running
  worker/image identity and active operator selection.
- [x] Keep diagnostics opt-in and bounded route counters on by default.
- [x] Verify W4A8 dispatch at 511/512/513 rows and real/mixed row errors.
- [x] Verify mixed padding, ordering, boundaries and capture fallback.
- [x] Record cold compilation and subsequent warm starts separately.

## C2 — Practical quality

- [x] Reuse the existing 30/12/6 regression tasks on the final image.
- [x] Freeze a held-out set of 48 work items from at least 12 source contexts:
  20 executable code/repository edits (at least eight multi-file or multi-step),
  12 reviews, eight exact retrievals and eight real structured/tool tasks.
- [x] Include at least eight long continuations or multi-step histories; at
  least two unused near-199K contexts and cold/warm prefix histories.
- [ ] Run candidate and frozen offline control on identical fixtures/histories.
- [ ] Pass every technical correctness/isolation/boundary check. No repeat of
  the known capped-199K page-review error, severe critical-task failure or
  concentrated new category failure. At most one additional noncritical
  failure relative to control; list every win/loss. Candidate solved count
  must be at least `control_passed - 1` if aggregate scoring is used.
- [ ] Record whole-prompt and affected-window NLL as diagnostics; never mark
  the old +0.0414 result as passing its former gate.

## C3 — Integrated serving and resources

- [ ] Complete the fixed 40-request trace over at least three fresh workers:
  12 C4 decode requests, six 32K+128K mixed requests, eight unequal C4,
  four near-capacity admission, six cancellation/recovery, four prefix-history.
- [ ] Include >32-shape cache churn/revisit, controlled single-request decode
  at 4K/32K/128K, sampled MTP serving and cold/warm compilation.
- [ ] Record TTFT, wall, output/queue times, bundle-gap distribution, route
  calls/tokens/reasons, host sync, preemptions and whole-device/Torch/cache
  memory. No unexplained growth, invalid access, hang or cross-request state.
- [ ] Preserve at least 20% summed wall improvement in 32K tasks vs R0 and
  128K tasks vs its W4A8 reference, and at least 20% long TTFT improvement
  in 32K+128K vs route-off on the final image.
- [ ] No reproducible >3% controlled median decode slowdown without an
  explicitly accepted cause. Target 32K+128K maximum bundle gap <=6.5 s;
  short TTFT penalty <=3 s with improved short completion.

## C4 — Freeze and deploy

- [x] Freeze image digest, loaded file hashes, policy hash, release tag and
  scorecard; fail startup on missing or incompatible required artifacts.
- [ ] Validate advertised text, reasoning, tool parsing, image and long-output
  API paths or narrow claims to what passes.
- [x] Set one systemd launcher/image and verify live service/worker identity.
- [x] Run the public source-review suite on the permanent service and the
  repeatable coding fixture on the same immutable image/policy; update README
  with current-only numbers and exact identities.
- [x] Review results locally before user-authorized remote publication.
