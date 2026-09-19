# E05 decision: stop at the profile/provenance prerequisites

All visible reference initialization kernels total 35.873 ms, or 0.617% of the
5810.889 ms sum of visible GPU kernel durations in the instrumented 8K request.
This includes necessary work outside GDN scratch and is broader than the
candidate region. It is not a critical-path percentage or a measured speedup.

The local native source checkout is known, but source equivalence to the
installed upstream wheel remains unproven. The review explicitly requires
native provenance and producer/consumer write coverage before removing zeros.
Neither should be assumed from matching symbol names or source appearance.

Given the small observed region and missing native equivalence proof, no native
scratch or Python `core_attn_out` initialization was changed. No `zeros` to
`empty` patch, no poisoning/State qualification of a candidate, and no serving
benchmark or promotion. Revisit only after a materially larger measured cost
and a reproducible native build have been established.
See [profile observations](../PROFILING.md).
