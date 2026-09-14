# Matched Q128 versus native Q256 at 196K

Both requests used the same 200,448-token prompt, 256-token output budget,
Batch 4,096, MTP4 with the full draft vocabulary, FP8 KV cache and a verified
180 W card limit. Both schedulers processed 217,088 prefill tokens, including
16,640 recomputed tokens from five preemptions, and both found all four
long-context markers.

| Logical checkpoint | Q128 time | Q256 time | Q128 time reduction | Q128 throughput | Q256 throughput |
|---:|---:|---:|---:|---:|---:|
| 96K | 126.115 s | 129.357 s | 2.51% | 779.48 tok/s | 759.94 tok/s |
| 128K | 185.115 s | 190.876 s | 3.02% | 708.06 tok/s | 686.69 tok/s |
| 192K | 390.115 s | 407.043 s | 4.16% | 503.97 tok/s | 483.02 tok/s |
| 196K end | 407.115 s | 425.490 s | 4.32% | 492.36 tok/s | 471.10 tok/s |

Full request wall time fell from 430.594 to 412.665 seconds (4.16%). The
high-resolution end-to-end TTFT is the most reliable comparison; intermediate
Q128 checkpoints came from one-second engine timestamps.
