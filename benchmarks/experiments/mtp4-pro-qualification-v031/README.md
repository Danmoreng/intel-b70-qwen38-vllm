# Bounded MTP4 comparison after Pro review

Only three frozen cells: source32K C1 / source64K C1 / prose32768 C4, weighted
35% /35% /30% **before** measurement. T1.0, top-p.95, top-k20,512 output tokens,
ignore EOS and fixed per-request seeds. Cold and warm prefix states are
separate strata. C4 salts differ between requests; cold/warm salts match within
an arm. Cache counters must establish the observed reuse.

Primary gain per paired block is the weighted mean of the three per-cell
native request-weighted decode-rate ratios minus1. Report cold and warm
separately, with equal weight only for a clearly labeled combined screening
number. C4 aggregate client decode throughput and per-request end-to-end/TTFT
are additional regression guards. Do not count tokens, concurrent requests,
or cold/warm repeats from one arm as independent ABBA blocks.

Use the same pinned vLLM0.31 / wide-M04 image for both depths,262144 tokens,
max_num_seqs16,4096 batched tokens and180W. Historical v0.30 pairs cannot
establish promotion after source/image drift. This is the authorized narrow
current pairing, not a repeated full baseline or another depth matrix.
`run.py` uses ABBA3/4/4/3, one untimed warmup per cell, then cold/warm pairs.
No power measurements are invoked by the copied streaming/accounting helper.

Promotion needs at least5% weighted real decode gain with positive paired
uncertainty, no important prose/C4 or latency regression, and unchanged memory
and correctness limits. Two independent paired blocks are screening evidence;
they cannot establish a precise confidence interval by resampling individual
tokens or concurrent requests as independent trials. An unclear result retains
the qualified MTP3 production image. Every exit restores and checks that image.

Completed ABBA result: combined weighted gain−.53%, descriptive95% interval
[−3.57%,+2.52%], with warm prose/C4 throughput regression. MTP4 does not clear
the promotion guards. All36 waves, including12 warmups, remain in receipts.
The immutable first helper divided512 outputs by the post-first interval;
derived analysis standardizes to511 outputs without changing any gain ratio
or rewriting raw records. Current helper uses the corrected numerator.
