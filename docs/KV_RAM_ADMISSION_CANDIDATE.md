# RAM KV cache and completion-reserved admission candidate

Status: isolated experiment, not a replacement for the qualified production release.
Base: immutable production EXL3 image `sha256:36e41ea027758060f9352bdd7f9ed47de5143d0424b238653960dbf8357c176a`, vLLM 0.31.0.

## Behavior

`runtime/kv_admission.py` supplies an `AsyncScheduler` subclass, preserving the
production engine's asynchronous pipeline. Before admitting
another request, it asks the native hybrid-cache coordinator how many additional
blocks every running or asynchronously loading request needs to reach its prompt
plus its allowed output and speculative lookahead. The new request is admitted
only if all those reservations fit. Already allocated blocks are deducted by the
coordinator. Async loads and active prefills are deduplicated. A five-percent
watermark applies when other requests exist; a single maximum-length request is
not blocked by that watermark.

This addresses the hole in full-input reservation: the native check can admit a
second large prompt while the first has allocated only its initial prefill chunk.
It also reserves output growth. Large requests wait in the native queue; short
requests can still overlap when they fit. Broad output limits intentionally make
admission conservative. This is a pinned-version adapter, not a portable vLLM API.

vLLM's `OffloadingConnector` stores reusable prefix state in CPU RAM and loads
matching prefixes back into GPU memory. The native hybrid groups include the
model's attention and recurrent state. This is prefix retention between requests,
not extra GPU memory for simultaneously decoding three full contexts. Active KV
states are not repeatedly swapped by this scheduler.

`runtime/xpu_cpu_offload.py` selects the upstream mmap-backed CPU buffer on XPU.
The native XPU per-tensor pinned allocator exceeded the logical cache budget on
this machine. The mmap region instead rounds the requested budget down to whole
chunks. It uses the upstream DMA path with ordinary host memory, without CUDA
host registration or CUDA-only UVA kernels. The candidate defaults to a **12 GiB cache
budget**, a 14 GiB `/dev/shm` ceiling, and a 23 GiB container memory limit with
swap disabled. The shm ceiling does not allocate 14 GiB by itself. Runtime,
metadata, drivers, and other services need RAM in addition to the cache budget.
The buffer's measured resident size is 12,867,993,600 bytes (11.984375 GiB),
with about 8 GiB host RAM still available during the full-context test. At a
10 GiB budget the same buffer measured 10,695,475,200 bytes (9.9609375 GiB).
These are `/proc/*/smaps` measurements, not Docker limits or virtual reservations.
GiB are binary units: the 12 GiB setting is approximately 12.9 decimal GB.
The first full-history restore nevertheless caused a host OOM: native H2D
copying also allocated a large pinned staging buffer. The corrected adapter
partitions load descriptors into at most 128 MiB payloads and waits for each
transfer to finish before submitting the next. Splitting without those waits
would still accumulate staging buffers. Stores retain the native path. A
standalone 9,175,040,000-byte transfer with a fully resident 12 GiB source buffer
passes data checks with a 14,684,524,544-byte container peak; the model-level
capacity test must pass separately. Large loads briefly block the worker during
the bounded copy; the native scheduler still manages admission and completion.

The CPU cache uses native ARC by default (`--cache-policy lru` is available for
a control). Favoring reused history over one-off insertions avoids simply
mirroring the GPU LRU. Rotation mode tests that distinction without a GPU reset.
The scheduler also reports GPU prefix reuse to ARC for ready CPU copies in
every hybrid group, including the required recurrent checkpoint. Native store
callbacks alone can omit reuse of that checkpoint: attention pages survive in
RAM, but cannot be resumed without their matching recurrent state. The adapter
does not invent cache entries, mark unfinished copies ready, or count a GPU hit
as a CPU transfer. This private manager interface is pinned to vLLM 0.31.0.

GPU and CPU prefix caches can contain overlapping copies. Their token capacities
must not simply be added to promise a combined context capacity. LRU admission
can retain duplicate prefixes and evict older ones from both tiers; a forced
GPU-cache reset demonstrates transfer correctness but does not prove a hit-rate
improvement for a real multi-agent history. A cache-policy/history study is needed
before generalizing that benefit to arbitrary agent histories with a CPU cache
this close to the GPU cache size.
Three simultaneous 262,144-token requests remain impossible with this GPU pool.
A static subagent limit alone cannot guarantee memory safety; actual token and
output budgets decide which model requests can overlap.

## Reproduction

Stop competing GPU workloads first. The launcher refuses to replace containers
or to launch while the normal production container is running. It never changes
the production image alias, systemd unit, `.env`, policy, or release manifest.

```sh
python3 scripts/run-kv-cache-candidate.py --cpu-gib 12 --test-endpoints
# Wait for http://127.0.0.1:8082/health to return 200.
python3 scripts/test-kv-cache-candidate.py --mode retention --tokens 16000 --count 1 --out benchmark-results/kv-ram-candidate/retention-16k.json
python3 scripts/test-kv-cache-candidate.py --mode admission --tokens 200000 --count 2 --out benchmark-results/kv-ram-candidate/admission-200k-c2.json
python3 scripts/test-kv-cache-candidate.py --mode rotation --tokens 200000 --count 2 --out benchmark-results/kv-ram-candidate/rotation-200k-c2.json
python3 scripts/test-kv-cache-candidate.py --mode cancellation --tokens 200000 --count 1 --out benchmark-results/kv-ram-candidate/cancellation-200k.json
python3 scripts/test-kv-cache-candidate.py --mode parallel --tokens 16000 --count 3 --out benchmark-results/kv-ram-candidate/parallel-16k-c3.json
python3 scripts/test-kv-cache-candidate.py --mode parallel-restore --tokens 16000 --count 3 --out benchmark-results/kv-ram-candidate/parallel-restore-16k-c3.json
python3 scripts/test-kv-cache-candidate.py --mode capacity --tokens 262000 --secondary-tokens 35200 --count 2 --out benchmark-results/kv-ram-candidate/capacity-262k-35k.json
python3 scripts/test-kv-cache-candidate.py --mode two-full --tokens 262000 --count 2 --out benchmark-results/kv-ram-candidate/two-full-262k.json
PYTHONPATH=scripts python3 -m unittest discover -s tests/unit -v
```

Retention mode resets only the GPU prefix cache on the otherwise idle test
server. It requires external-cache hits and identical deterministic output,
including three secret records present only in the cached history. Transfer
metrics preserve CPU-to-GPU and GPU-to-CPU labels. Admission mode requires a
waiting request, at most one active request, and no added preemptions. Parallel
mode forces 256 output tokens to observe sustained concurrent decode rather
than only a seven-token answer. Cancellation mode disconnects a live long-prefill
stream, requires all scheduler work to clear within 30 seconds, and verifies a
subsequent small request completes.
Rotation mode first reuses a 16K history with three secret records, then submits
two distinct 200K requests concurrently, and finally asks for that original
history again. It requires a real RAM reload after natural GPU eviction,
unchanged secret-record retrieval, at most one active pressure request, a
waiting request, and no preemptions. Host RAM availability is sampled too.
Parallel-restore mode clears GPU prefixes after seeding three independent
histories, then restores them concurrently. Capacity mode warms each history
three times, clears only the GPU cache, and requires at least 290,000 distinct
prefix tokens across two histories to return from RAM. Every history must show
a completed CPU-to-GPU transfer, zero GPU prefix hits, and correct secret records
from its beginning, middle and later context. A partial final cache block is
recomputed. This is a retention-capacity test with repeated histories, not a
promise that an arbitrary one-off request immediately populates the same useful
capacity or that GPU and CPU contain entirely different histories.
Two-full mode runs two independent nearly maximum-size histories and then
returns to each of them in turn, without resetting the GPU cache. It requires
a nearly complete cache hit at each admission, identical records and no
preemption. An insufficient prefix hit cancels that request immediately, rather
than waiting for another full cold prefill. It can reuse the first history
warmed by capacity mode. This tests one round of full-history alternation,
not indefinite alternation between arbitrary changing histories.

Use `--native-admission --no-offload` to run a control on the same immutable
image. Test endpoints are opt-in and the candidate binds localhost only. Stop
and remove the candidate before using the qualified service again:

```sh
docker stop -t 30 b70-kv-ram-candidate
docker rm b70-kv-ram-candidate
```

Raw launch, timing, output hashes, queue samples, and transfer metrics are in
`benchmark-results/kv-ram-candidate/`. A committed assessment records the tests
that actually completed; commands above are not claims of completed validation.
The initial synchronous prototype's results are retained under `sync-v1/` and
must not be presented as qualification of the corrected asynchronous adapter.
`async-v2/` records intermediate asynchronous/LRU tests; `async-arc-v3/`
records ARC before the hybrid GPU-hit fix. Its natural-eviction restoration
failed and must not be represented as a passing cache test. Its two 200K input
requests did correctly queue without preemptions. `hybrid-v4-scaled/` verifies
the corrected hybrid reuse policy under natural eviction with a reduced GPU
pool: two 64K requests queue, and the old 16K history returns from RAM with all
secret records intact. Three concurrent RAM restores also pass there.
`hybrid-v5-full/` records the corrected full-GPU candidate with a 12 GiB budget.
It failed on the first full RAM reload due to the additional staging allocation.
`hybrid-v6-bounded-load/` adds bounded staging and records the requalification.
`native-control/` uses the same
image without offloading or the admission adapter. These exploratory single
performance comparisons do not constitute a statistical release benchmark.

In this vLLM version `cpu_cache_usage_perc` measures in-flight transfer pinning,
not retained cache occupancy. An idle value of zero does not mean the RAM cache
is empty. Completed CPU-to-GPU transfer bytes and external prefix hits provide
the relevant restoration evidence.

## Production considerations

This experiment intentionally leaves the existing qualified release unchanged.
Production adoption needs a separate attested profile/launcher and matching
runtime hashes, cancellation/recovery and restart checks, realistic alternating
agent-history measurements, and comparison against the same-image native
control. A companion cockpit patch replaces the application relay's hardcoded
120-second idle timeout with a validated, configurable 30-minute default
(`B70_RUNNER_INFERENCE_IDLE_TIMEOUT_MS`, bounded to one hour), while preserving
task cancellation and capability revocation. It is prepared in the isolated
cockpit checkout and is not installed into the productive runner by this test.
Queued or very long cold-prefill requests can legitimately produce no tokens
for several minutes; a production switch must include that companion patch.
The native queue order is retained, including possible head-of-line blocking
of a small request behind a large one that does not currently fit.

Official reference: https://docs.vllm.ai/en/v0.31.0/features/kv_offloading_usage/
