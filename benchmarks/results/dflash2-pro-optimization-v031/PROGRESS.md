# Pro follow-up: completed evidence, no serving promotion

The bounded work order is complete. See [the final decision](DECISION.md) and
[all matched panel metrics](panel-comparison.json). Both DFlash projections
fail the performance/C4 guards; MTP3 remains the general serving choice.

The first infrastructure-only commit is `d857bd1`. Production retains the
qualified vLLM0.31 MTP3 image, FP16, FP8 KV,262144 tokens and180W. No new power
matrix is run. Existing per-wave energy counters are incidental measurements;
the README power comparison remains historical.

## Completed DFlash screens

All initial speed cells use the frozen source-review prompts, three measured1024-token
repeats after an excluded warmup, T1.0, top-p.95, top-k20 and the original seeds.
Actual prompt lengths are32722 and65491. Medians use native request-weighted
decode time. All individual repeats and acceptance counters are retained in
`initial-screens.json`; historical production numbers are not paired controls.

| Candidate | Configured context |32K C1 tok/s|64K C1 tok/s|
|---|---:|---:|---:|
| FP16 draft, probabilistic proposals |96000|54.04|46.08|
| W4 draft, original grouping |96000|56.44|47.75|
| W4 draft, grouping four |192000|55.90|52.54|
| W4 draft, grouping four, K/V-only context |192000|59.51|45.74|

The original W4 grouping rejected192K at startup (8.92GiB required,
7.94GiB available); its166400 estimate is not an actual capacity qualification.
The converted draft contains1280616960 logical parameter bytes, approximately
1.193GiB. Its21 packed runtime modules cover36 checkpoint members and passed
independent FP64 selected-column dot products spanning every fused shard.
The shared full target head and embeddings remain aliases, and no dense context
KV weight copy is kept. The draft backend is oneDNN W4A16; the target is EXL3.

Grouping four has now passed an actual190976-token input plus1024 outputs:
192000 total, no preemption, no unexpected recomputation. A separate near-limit
191984-token input with16 stochastic output tokens completed, and a warm
custom-token stop returned exactly one output token. These are capacity/state
smokes, not a semantic quality score or full sampling-law qualification.

The initial capacity-test group-four pool holds627 blocks at13MiB per block. At192K the
CPU reconstruction needs588 blocks (7.465GiB), with about.076GiB padding,
versus562 original16.25MiB blocks (8.918GiB), with about1.530GiB padding.
The live-layer/reservation requirement is unchanged. Pool utilization improves;
the physical allocation in that capture is about7.96GiB. Available profiled
pool capacity differs between diagnostic launches; it is not a fixed hardware
constant. The final panel separately records its actual allocations.

The initial grouping speed screen also changed max context96K→192K. Do not
attribute its64K median increase solely to grouping. Stochastic outputs differ
and acceptance varies substantially, including the original grouping's68.74
tok/s third64K repeat. That high sample is retained, not reported as typical.
Matched-context and deterministic controls are reported separately below
for the grouping speed/state comparison.

## Bounded MTP outcome

All four ABBA arms completed and restored the exact production image. Across
the predeclared35/35/30 mix, MTP4 gains+.11% cold and-1.16% warm, or-.53% with
equal cold/warm weight. The descriptive paired95% t interval for this combined
screen is[-3.57%,+2.52%], based on just two independent AB/BA blocks. This is
not a precise general workload confidence claim.

Warm prose/C4 aggregate client decode regresses19.15% and17.43% in the two
blocks; mean client end-to-end time rises19.74% and22.09%. Long C1 rates are
higher with MTP4, but this mix fails the5% gain and C4 regression guards.
Production therefore retains MTP3. No further MTP depth matrix is warranted
by this bounded outcome. `mtp4-bounded/` retains all36 waves, including12
excluded warmups, the original source hashes and exact restoration receipt.

Native decode rates exclude the first output per request. The initial copied
MTP helper used all512 outputs over that interval; the derived analysis and
public receipts correct absolute rates to511 (-.1953125%). Every paired rate
ratio and the decision remain unchanged. Original raw records are preserved.

Eight real greedy and eight probabilistic unquantized DFlash captures also
pass independent scalar Philox acceptance-prefix replay. The CPU Philox
oracle passes three published Random123 known-answer vectors. Actual residual
and bonus Gumbel winners pass in the newer greedy captures, including both
W4 matched-grouping arms. The older probabilistic captures lack the attested
RNG mode and do not claim proposal/residual/bonus winner coverage.

## Additional scoped checks

- Separate packed K/V-only context projection, preserving the exact K/V
  coefficient storage and normal B8 query QKV. All five loaded K/V projections
  match the full packed QKV output exactly in the eight-row probes; independent
  FP64 packed-dot probes still cover all21 modules. Its actual192K and context-end gates passed. The speed medians59.51/45.74
  tok/s have large trajectory-dependent variance; no stable advantage is claimed.
- Actual eager feature/branch/norm/conv/selector/RoPE/stored-KV capture: four
  decode rounds passed244 numerical comparisons, plus22 small-input epsilon
  norm probes and eight actual acceptance/residual/bonus RNG replays.
  The greedy C4 rejection kernel also passed changed-slot graph replay.
  `actual-eager/` preserves the failed capture/reference attempts. Ideal FP64
  RoPE rotations exceed the original bounds at32K; actual cache-theta and
  native rotations with the stored FP16 coefficients pass those same bounds.
  Ideal rotation errors are retained, not relabeled as a PASS. Graph-internal
  branch captures and a broad semantic qualification remain limitations.
- Same96K-limit group-five/group-four controls completed: all four256-token
  greedy cold/warm outputs match exactly across groups, with observed warm
  cache hits. Group-four greedy rates are.34/.80/.37/2.48% lower; the largest
  difference primarily reflects86 versus84 proposal rounds. Stochastic32K
  medians55.75→55.51 tok/s and64K45.87→49.11 are separate trajectory-dependent
  observations, not isolated kernel gains. Each arm passes eight independent
  acceptance/residual/bonus RNG replays and exact production restoration.
  `grouping-matched96.json` retains all values and caveats, including a short
  CPU-only replay overlapping the group-five64K second repeat.
- Both full-QKV and K/V-only DFlash candidates completed the same512-output,
  three-cell panel as the MTP study. Weighted native gains are−5.56% and−4.97%;
  warm C4 aggregate client decode regresses41–43%. These are observational
  comparisons against earlier MTP controls, not a paired promotion study.
  Registered draft weights/buffers occupy1.255/1.256GiB, excluding shared target
  weights and separate speculator/KV/graph allocations. Actual scheduler logs
  periodically show Running3/Waiting1 despite configured max_num_seqs16.
  Each arm restored and verified the exact qualified production image.

On October9 the user permitted about166K total tokens if DFlash has a clear
speed advantage.192K stays preferred. This supersedes the attachment's hard
192K minimum; it does not waive correctness or speed evidence.
