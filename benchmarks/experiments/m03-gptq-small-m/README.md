# M03: GPTQ small-M

The first challenger keeps the loaded symmetric GPTQ W4/G128 coefficients and
FP16 activations unchanged. A narrow SYCL kernel reuses each decoded weight
across M=1..5 rows. The initial promotion gate is only the highest-cost
gate/up family K=5120, N=34816; all other shapes and every M>5 retain oneDNN.

Correctness is compared against the executing `_xpu_C.int4_gemm_w4a16` backend.
Performance uses five warmups and 31 alternating timed calls per arm. Serving
integration is forbidden unless M=1,3,5 are all correct and each is at least
3% faster.
