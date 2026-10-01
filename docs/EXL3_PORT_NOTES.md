# EXL3 migration execution notes

The authoritative scope is [the Pro implementation plan](EXL3_PRODUCTION_MIGRATION_AND_OPTIMIZATION_PLAN.md).
The active migration branch is `work/exl3-migration-20261001`; the EXL3 worktree is
`../exl3xpu-migration`. GPTQ remains the production release. No production tag or
launcher has been changed. Release requires all gates and explicit acceptance
of any remaining performance regression.

## Baseline — 2026-10-01

`benchmarks/results/exl3-migration/baseline_environment.json` captures source,
runtime, checkpoint and loaded native-library identities. Original benchmark
JSON is copied byte-for-byte into `baseline-evidence/`; these are historical
results, not measurements of the migration branch. The EXL3 image was inspected
without starting a GPU worker; its complete loaded-library map is still required
when the production-contract candidate runs.

The two vLLM trees in the GPTQ image differ. The `vllm` console script resolves
`/opt/venv/lib/python3.12/site-packages/vllm`; `python -c` from the image working
directory resolves `/workspace/vllm/vllm`. Workers inherit the console script's
search path. Tests and patches must identify the launch path rather than assume
that an interactive import describes deployed serving. The container command,
console search path and hashes of both code trees are recorded.

The pinned production image is
`sha256:ed1ebca756abb0e0832d11cd0db026dd7e86df094c6903efe7ae8afbdc290b68`.
The old EXL3 image is
`sha256:cba73584f4ab0a2b37eac1356f34f16655ac5e740d845e110997b03be78279b7`.
Power was verified at 180 W. Runtime versions remain Torch 2.13/vLLM 0.30 for
GPTQ and Torch 2.12/vLLM 0.26.1 for the initial EXL3 baseline.

## Safe foundation

EXL3 compilation now uses a private temporary library, full compiler logs,
`pipefail`, registration/capability smoke tests and atomic installation. The
Torch C++ ABI is obtained from the build environment. The adjacent manifest
records compiler, flags, sources, Torch, driver/IGC, oneDNN and library hashes.
An actual invalid `icpx` flag failed with exit 1 and preserved both the previous
library and its manifest byte-for-byte. The successful Torch-2.12 build exposes
all 18 expected operators. This library must not be reused for Torch 2.13.

The loader and bit-exact test share an authoritative `.trellis` inventory. The
checkpoint contains 409 quantized modules: 408 at 4 bpw, one target LM head at
6 bpw, including eight MTP modules absent from `tensor_storage`. Missing
components, contradictory formats/codebooks and duplicate canonical names fail
early. EXL3 configuration rejects all three GPTQ draft overrides. The separate
GPTQ runtime patch selects overrides only for GPTQ formats and rejects an INT4
MTP conversion without unquantized source weights. The existing production
combination (BF16 construction followed by INT4 draft conversion) is supported.

Native results in `benchmarks/results/exl3-migration/safe-foundation/`:

| Check | Result |
|---|---|
| Bit-exact reconstruction and one-hot GEMM | PASS, 409/409 modules; 8/8 MTP; no skips |
| INT8 prefill, M=8192, independent Torch INT8 reference | PASS; maximum relative error 0.0002695601, unchanged tolerance 0.002 |
| Deliberately perturbed INT8 reference | Expected FAIL/exit 1; maximum relative error 0.0909117 |
| Deliberate actual compiler error | Expected exit 1; installed library and manifest unchanged |
| GPTQ restoration after exclusive GPU tests | PASS; original pinned image, HTTP health 200 |

These tests prove the tested native reconstruction/linear paths. They do not
qualify the new loader on a complete serving model, attention, the 0.30 port,
FP8 cache quality, graphs, long-context capacity or throughput.

Next: complete loader reports, qualify the existing EXL3 image under the
200,704-token/C4/media/API contract, and instrument scheduler/cache allocations
before diagnosing preemptions. The runtime port and performance experiments
follow those measurements, in the order specified by the plan.
