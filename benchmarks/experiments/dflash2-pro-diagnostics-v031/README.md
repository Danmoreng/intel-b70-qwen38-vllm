# Pro review diagnostics, vLLM 0.31.0

Authorized work order: `PRO_ASSIGNMENT.md`, copied unchanged from the user attachment.
Production remains FP16/MTP3/FP8-KV/262144 tokens/180 W until qualification.

Execution order:

1. A0: attest the actual B8/V7 worker and serialize the real cache allocation,
   per-group page/block requirements at 96000 and CPU-modeled 192000 tokens.
   Report allocator residency separately from logical tensor sizes.
2. A1: independently audit proposal/rejection distributions. Capture selected
   real graph-replay rounds separately from timing. Dense and paged draft
   attention references are required before any DFlash promotion.
3. A2: time runner call sites after graph capture, then collect a bounded XPU
   kernel trace. Event spans include dispatch gaps; they are not kernel sums or
   an uninstrumented throughput benchmark.
4. First commit contains infrastructure and receipts only. Decide subsequent
   changes from measured costs, preserving rollback and one variable per arm.
5. B8/V7 no-op separation, then B8/V4 only if expected savings justify it.
   W4A16 role-aware draft loading and real 192000-token qualification are a
   separate track. MTP4 qualification is limited to long C1 and prose/C4 cells.

Predeclared numerical reporting: selector cache scores must agree exactly;
temperature-normalized q TV <= 1e-6 for identical realized scores; independent
FP32 rejection identity <= 1e-12. Numerical attention tolerances will be frozen
in the attention fixture before its GPU evaluation. Statistical tests use
familywise alpha 0.001 and report sample size and confidence bounds.

No production defaults, qualified reference checkouts, power comparisons, or
previous baseline results are rewritten by these diagnostics.

`run_branches.py NAME` is an additional perturbing **eager** diagnostic pass:
actual post-residual feature taps, FC/KV projections, norms, convolution,
selector, RoPE and stored paged KV are captured on four selected decode rounds.
`replay_branches.py` uses independent CPU arithmetic and frozen error bounds.
Inputs to in-place norms and residual buffers are cloned before subsequent
mutation. This pass does not establish graph-internal branch equivalence and
must never be used for throughput comparisons. The controller stores immutable
source snapshots and verifies the exact qualified image after every exit.

`rng_reference.py` is a scalar Philox/Murmur/Gumbel oracle independent of the
GPU implementation. It passes three Random123 known-answer vectors and
replays discrete decisions only with explicit threshold/winner margins. Old
captures lacking the attested Gumbel mode retain acceptance-prefix coverage
without claiming proposal/residual/bonus draw replay. `publish_branches.py`
exports the newer actual eager evidence, including failed attempts.

The eager RoPE diagnostic uses two explicit checks at unchanged bounds:
independent ideal theta versus the cached coefficients, then the rotation
kernel versus those actual FP16 coefficients. Ideal FP64 full-rotation errors
are reported separately and exceed the original bounds at32K. This separation
does not establish ideal-FP64 equivalence or graph-internal branch equivalence.
