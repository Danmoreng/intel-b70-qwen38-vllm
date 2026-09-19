# Profile observations for the conditional follow-ups

Evidence: E01 serving study `serving-20260919-200901`, block 0, one excluded
8K-prompt/32-output profiler request per arm after graph warmup. The timing
comparison uses separate uninstrumented requests. `profile-summary.py` extracts
the following evidence from the recorded traces.

The reference trace exposes prefill device kernels, but not the interiors of
decode graph replays. Summed kernel durations are not critical-path times and
cannot be presented as end-to-end speedup estimates. In particular, absence of
an M04 reducer in this trace does **not** establish that it is free or unused.

## E01 dispatch

The reference has zero calls to `_fused_qk_rmsnorm_rope_gate_kernel`; the candidate
has 51, consistent with three prefill chunks across 16 target attention layers
and one draft layer. The candidate dispatch is therefore active. Its isolated
operator gain must still pass the serving comparison: the deployed baseline is
compiled and already fuses work that the eager operator reference keeps separate.

## E03 measurement needed

`e03/layout-cost.py` measures the whole existing M04 region, an intentionally
incorrect diagnostic with the output conversion omitted, and output conversion
alone. It uses the production binary and fixed split policy for M=2,3,4,5 at
8K/16K/64K/196K, with paired inputs and eager/graph measurements. The elision arm
only estimates removable work; it is never a candidate implementation.

## E04 prerequisite not established

The reference trace contains norm kernels such as
`triton_red_fused__to_copy_add_fused_add_rms_norm_4`. The actual TorchInductor
source loads the weight as FP32 and adds 1 inside the norm kernel before its
final multiplication/store. `e04/compiled-evidence.json` retains representative
generated definitions and source hashes. There is no observed independent
weight-conversion/add launch to remove. This proves fusion, not that the inline
addition has literally zero cost. A weight-buffer cache is not justified by the
current evidence; decode kernel interiors remain a profiler limitation.

## E05 small observed region and unresolved native provenance

In the reference, **all** visible initialization kernels together account for
35.873 ms of the 5810.889 ms sum of visible GPU kernels (**0.617%**). This includes
native FillFunctor, compiler-generated zeros, and KV zeroing: it is broader than
the proposed removable GDN scratch work, and some of it is semantically required.
The candidate trace is similar (35.851 ms, 0.616%). These are profiler observations,
not measured savings.

The local native source checkout is `6d92b1bfbf32767ecda8e819613eb151e70030ad`.
The export's `native-sources/vllm-xpu-kernels/SOURCE-PROVENANCE.json` explicitly
does not assert equivalence between that checkout and the installed upstream
wheel. No new source-to-wheel proof has been established. The review's required
conditions for changing native GDN scratch initialization are therefore unmet.
Do not replace zeros with empty based on source appearance alone.
