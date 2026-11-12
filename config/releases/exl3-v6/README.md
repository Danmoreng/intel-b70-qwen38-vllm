# Previous MTP3 MB4/K4 production profile

Frozen v6 image and policy from commit `a48c855`, with archived RAM/LBNHC modules. This retains the original1600/1600 cache geometry without the new fine-prefix COW wrapper. Stop the current service before starting `scripts/run-server-exl3-v6-rollback.sh`; use `--check-only` to validate without loading the GPU. The current12GiB host RAM budget and180W cap remain.
