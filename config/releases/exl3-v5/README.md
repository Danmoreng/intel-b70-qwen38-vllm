# Previous vLLM 0.33 MTP3 production profile

This is the unchanged pre-matrix image `sha256:f1822d3b533eb77a3d9f9c3b610bab0c8334c329a40d6ed32041345d79ad567e`, with its frozen policy and independently archived 12 GiB RAM/LBNHC runtime modules. The original public source state is commit `4e362ae`. Native/Python source and qualification remain pinned in the preceding release.

Stop the current service before rollback. `scripts/run-server-exl3-v5-rollback.sh --check-only` validates the archived profile; without that option it starts the old engine. Both use the same GPU and API port.
