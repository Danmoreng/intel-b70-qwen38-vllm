# Compact MTP depth screening

The user requested a smaller and faster MTP3/MTP4 comparison on 2026-10-01.
The original 160-wave ABBA campaign was intentionally curtailed after 31
completed MTP3 waves. Its raw outputs, original controller source, interruption
record and completed measurements are retained in
`benchmark-results/exl3-mtp-serving-v1`.

The compact screen uses 18 waves per depth, 36 total:

- Code and prose at 4,096 and 49,152 tokens: C1/C4, cold/warm (16 waves).
- Code at 102,752 tokens: C1, cold/warm (2 waves).
- MTP3: reuse the exact 18 completed waves from the curtailed campaign.
- MTP4: one fresh matched pass, 512 output tokens per request.

The runtime image, tokenizer inputs, serving settings, output budget and power
cap remain identical. The controller verifies the frozen panel and image IDs,
all settings, each persisted/raw-result equality, prompt verification and output
counts. Reused results retain their source paths and SHA-256 hashes.

This is a single-pass screening comparison, **not a completed ABBA experiment**.
The historical MTP3 pass does not control order drift. Repeat only ambiguous or
contradictory cells before selecting a depth. Long C4 pressure, full-context
qualification and generated-output correctness remain separate release gates.

CPU validation passed: 18 compact cases per depth, original 40-case-per-arm
selection preserved, all 18 existing MTP3 results accepted, wrong image/panel/
settings and an unfinished source wave rejected. Python compilation and
`git diff --check` passed. The compact XPU campaign is in progress; no depth
selection or performance improvement is claimed yet.

The previously queued broad component diagnostic was cancelled before GPU
execution. Profiling will be narrowed to the cases needed to explain the compact
comparison. GPTQ stays offline.
