# Bounded generated-path diagnostics — complete

Four fresh routes on the same pinned image/profile: graph-MTP3, graph-MTP4, eager-MTP3 and graph-nospec. Each has eight4K code/prose C1/C4 cold/warm waves and four identical-prefix one-token controls. All96 measured requests complete; all10,240 main output tokens choose a reported maximum-score token.

Against graph-MTP3, exact output matches are7/20 for MTP4,5/20 for eager and9/20 for nospec. First-divergence reported top1/top2 margins reach0.21875 logprob units; these are not uniformly tiny ties. On four identical-prefix controls, graph-MTP4 chooses the same next tokens, while eager flips code137 and nospec flips prose3. Those controls contain prefill target calls, so the flips do not establish a speculative-verification-only fault.

Top5 scores do not establish full-vocabulary KL or BF16 generated-path equivalence. Gate E remains partial; do not call all differences harmless rounding or a sampler bug. Exact output IDs, scores and metadata remain in the raw campaign; assessment.json retains audited hashes and comparisons. No new BF16 references or engine/kernel changes were made for this diagnostic.

MTP3 is now the target-profile default for the next optimization experiments. The next serving ablation compares65536-row pruning with the full draft vocabulary at fixed MTP3, rather than reopening depth selection.
