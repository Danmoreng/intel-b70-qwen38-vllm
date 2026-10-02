# EXL3 v2 power-limit comparison — 2026-10-02

Same immutable EXL3 v2 image, serving arguments, environment values, frozen
prompts and sampling. The existing fresh 180 W matrix is the baseline; 230 W
and 275 W each repeat **20 scenarios / 70 measured waves / 124 requests** plus
an isolated 64K cold/warm resend. All ordinary requests completed without
preemptions. The production service is restored to **180 W**.

These are measured hardware variants; they do not change the qualified
production policy. Power limits are not assumed actual consumption: the table
uses the card energy counter. Measured wave times exclude warmup/startup;
energy includes each measured wave's prefill, decode and request overhead.

| Power limit | Measured mean card power | Wave time | Output tok/s | Output tok/s per W | Output tok/Wh | Card energy |
|---:|---:|---:|---:|---:|---:|---:|
| 180 W | 180.1 W | 41.68 min | 50.8 | 0.282 | 1,015.0 | 125.10 Wh |
| 230 W | 230.1 W | 35.77 min | 59.2 | 0.257 | 925.7 | 137.16 Wh |
| 275 W | 275.0 W | 33.89 min | 62.4 | 0.227 | 817.4 | 155.34 Wh |

All efficiency columns cover the same **126,976 output tokens**, including
their prefill and request overhead. `Output tok/s per W` divides whole-wave
throughput by measured mean card power: it equals output tokens/J. `Output
tok/Wh` is output tokens divided by integrated card energy; **higher is
better**. These are not decode-only rates, and do not include whole-PC power.

For this fixed mixed workload, **180 W produced the most tokens per Wh**;
**275 W finished the measured waves fastest**. The rate tables below show
which context and concurrency points benefit from the additional power.

| C1 input budget | 180 W prefill / decode | 230 W prefill / decode | 275 W prefill / decode |
|---:|---:|---:|---:|
| 4K | 2,241.4 / 57.1 | 2,615.2 / 66.8 | 2,791.5 / 69.5 |
| 16K | 2,168.2 / 56.2 | 2,540.1 / 65.5 | 2,721.1 / 68.4 |
| 64K | 1,670.0 / 52.7 | 1,947.6 / 60.6 | 2,086.1 / 63.9 |
| 128K | 1,262.5 / 39.9 | 1,466.3 / 45.9 | 1,562.6 / 48.0 |
| 200K | 999.3 / 38.5 | 1,154.4 / 44.0 | 1,228.2 / 46.2 |

| Input per request, C4 | 180 W aggregate decode | 230 W aggregate decode | 275 W aggregate decode |
|---|---:|---:|---:|
| 2K | 190.5 | 223.5 | 231.1 |
| 4K | 180.1 | 204.7 | 219.0 |
| 16K | 165.0 | 191.5 | 202.2 |

## Complete matrix

Each cell is median native prefill / request-weighted decode / weighted MTP
acceptance. Rates are tok/s; C4 request-weighted decode is not aggregate decode.

| Scenario | 180 W | 230 W | 275 W |
|---|---:|---:|---:|
| phase-512-c1 | 1718.8 / 58.4 / 55.1% | 1928.4 / 68.3 / 55.1% | 1969.0 / 71.1 / 55.1% |
| phase-2k-c1 | 2431.6 / 59.9 / 59.5% | 2830.3 / 70.2 / 59.5% | 3003.9 / 73.2 / 59.5% |
| phase-4k-c1 | 2241.4 / 57.1 / 55.0% | 2615.2 / 66.8 / 57.5% | 2791.5 / 69.5 / 56.4% |
| phase-8k-c1 | 2305.6 / 54.9 / 52.2% | 2712.2 / 64.2 / 52.2% | 2902.4 / 67.0 / 52.2% |
| phase-16k-c1 | 2168.2 / 56.2 / 56.3% | 2540.1 / 65.5 / 56.3% | 2721.1 / 68.4 / 56.3% |
| phase-32k-c1 | 1977.7 / 54.5 / 56.6% | 2310.0 / 65.3 / 59.7% | 2473.1 / 65.1 / 55.7% |
| phase-64k-c1 | 1670.0 / 52.7 / 60.7% | 1947.6 / 60.6 / 60.7% | 2086.1 / 63.9 / 60.7% |
| phase-128k-c1 | 1262.5 / 39.9 / 54.5% | 1466.3 / 45.9 / 54.5% | 1562.6 / 48.0 / 54.5% |
| concurrency-2k-c2 | 1641.6 / 53.2 / 56.0% | 1918.0 / 62.0 / 57.2% | 2051.6 / 65.0 / 56.5% |
| concurrency-2k-c3 | 1092.7 / 50.7 / 59.9% | 1283.9 / 59.0 / 59.9% | 1377.2 / 61.1 / 60.1% |
| concurrency-2k-c4 | 1002.6 / 45.9 / 59.0% | 1178.0 / 53.5 / 59.9% | 1266.3 / 55.6 / 58.9% |
| concurrency-4k-c2 | 1405.7 / 51.9 / 56.9% | 1642.3 / 61.5 / 56.8% | 1766.1 / 63.3 / 57.0% |
| concurrency-4k-c3 | 1130.8 / 43.8 / 53.4% | 1329.6 / 50.9 / 53.7% | 1435.0 / 56.0 / 56.2% |
| concurrency-4k-c4 | 1024.6 / 41.7 / 54.9% | 1205.9 / 47.4 / 54.7% | 1300.5 / 50.9 / 56.9% |
| concurrency-16k-c2 | 1671.4 / 44.3 / 58.3% | 1956.3 / 52.1 / 58.1% | 2099.2 / 54.4 / 59.2% |
| concurrency-16k-c3 | 1546.8 / 34.9 / 57.8% | 1809.2 / 41.7 / 57.3% | 1940.7 / 42.3 / 57.7% |
| concurrency-16k-c4 | 1484.9 / 29.9 / 59.1% | 1733.0 / 34.5 / 59.3% | 1861.3 / 36.6 / 60.5% |
| prefix-16k-cold-warm | 2026.0 / 62.3 / 66.6% | 2305.0 / 73.8 / 66.6% | 2433.0 / 77.0 / 66.6% |
| prefix-64k-cold-warm | 1269.2 / 49.7 / 57.8% | 1482.0 / 57.3 / 57.8% | 1567.0 / 60.2 / 57.8% |
| full-context-199680 | 999.3 / 38.5 / 61.2% | 1154.4 / 44.0 / 61.2% | 1228.2 / 46.2 / 61.2% |

## Temperature and attribution

- 230 W: maximum sampled package 72.0 °C / VRAM 84.0 °C; 1479 hardware samples.
- 275 W: maximum sampled package 73.0 °C / VRAM 86.0 °C; 1405 hardware samples.

The 180 W baseline has no comparable thermal trace. One campaign per limit
with the fixed repeats is a finite observation, not a randomized confidence
interval. Different output trajectories/acceptance and thermal histories can
affect results. Energy covers only the same 70 measured waves; supplementary
prefix requests, warmup/startup and whole-PC energy are excluded. No coding
campaign or quantization quality study was repeated. Both power settings were
accepted and verified without sudo; the controller restored the original
180 W cap and healthy production service.

Image `sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918`; policy `1bf624713cff3583148a71c3b4fb9189268cc5d4a47afca03662ce25be3e7839` still describes
180 W production. [Compact comparison](../benchmarks/results/exl3-power-20261002/comparison.json),
[230 W summary](../benchmarks/results/exl3-power-20261002/230w-serving-summary.json),
[275 W summary](../benchmarks/results/exl3-power-20261002/275w-serving-summary.json) and
[final controller/recovery receipt](../benchmarks/results/exl3-power-20261002/campaign.json) preserve identities.
