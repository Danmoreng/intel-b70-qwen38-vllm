# Previous MTP3 fine-prefix production profile

Frozen v7 image and policy from public commit `1a8dfe004b3dce5e5b2995b5eba54f64e8c17619`, with its archived RAM/LBNHC runtime modules. This retains PyTorch 2.14, 1792 GPU cache blocks, the 128-token prefix block and the dense-alias copy-on-write helper. The host cache budget is 12 GiB and the GPU power cap is 180 W.

Stop the current service before starting `scripts/run-server-exl3-v7-rollback.sh`; use `--check-only` to validate the saved release without loading the GPU.
