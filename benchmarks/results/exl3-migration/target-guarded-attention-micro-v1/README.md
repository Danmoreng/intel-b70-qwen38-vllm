# Unified guarded attention: standalone correctness passed

The source-guarded V2 metadata adapter uses exact CPU lengths only for genuine
prefill rows. Decode retains actual GPU lengths, with CPU upper bounds used
only for launch capacity. Adaptive verification and unsupported semantics fall
back unchanged. Contiguous native groups use true per-request maxima, without
assuming a decode-first ordering or reading GPU lengths on the host. Needed
relative query boundaries are prepared once per metadata build. The compute
hook does not write KV; native V2 retains that responsibility.

31 XPU cases pass the pre-existing rtol0.01/atol0.003 contract, covering
q1–5/63/64/255/256, KV4095/4096/4097 and32769/102753, C4 interleaved
requests, page64/1600/1664 and nonunit FP8 scales. Maximum absolute error
versus native is0.0001221 and versus independent FP32 is0.00008290. KV
bytes stay unchanged. Two graphs replay correctly after query/page/length
changes; oneDNN never enters capture. Three future-KV poison cases leave the
first query bit-identical while changing the last query. Five CPU contract
tests cover malformed/absent metadata, thresholds and semantic fallback.

The library is now packaged with source and manifest in a separate immutable
candidate image, with a build-time hash/ABI/operator import check. Real mixed
serving, numerical quality and full release qualification remain mandatory.
