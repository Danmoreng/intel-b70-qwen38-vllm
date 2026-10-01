# Final candidate shape and mixed-serving evidence

Immutable runtime6cf48999, native library48f879c1, M04 eaa18427.

Mixed ABBA changes only guarded prefill; MTP3/M04 remain enabled. Eight waves/24 requests complete with exact native accounting, no preemption and no cache hits. Cold49K incoming-prefill TTFT improves~24–25%, overlapping SSE burst-gap p95~31–33%. These gaps are client-observed bursts, not GPU token-step timings. API/tools/reasoning/media limits pass. The32 maximum-area images here share a hash; independent images are a separate final operational gate.

All409 weight tensors including8 MTP tensors reconstruct bit-exactly;26.02 billion weights checked. Seven linear classes cover18 row counts1–512, all248320 output-head columns, finite independent reference checks at the unchanged RMS2e-3 tolerance, tail poisoning, compiled large-first ordering and mutated graph inputs. Complete passed body classes were reused after a later harness Dynamo closure specialization limit; all head classes were rerun with isolated compiler state. Failed or incomplete subcases are excluded.

Frozen production graphs cover at most64 decode rows. SmallM graphs additionally pass through128. Large INT8-prefill rows129/256/512 are explicitly eager/compiled only: capturing that oneDNN prefill primitive is unsupported and outside FULL_DECODE_ONLY. No unsupported graph-capture failure is relabeled a pass. Raw source receipts are in rows.json.
