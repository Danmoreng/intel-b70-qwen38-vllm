# M00 baseline decision

The immutable control is the promoted Q128, 4096-token scheduler chunk, MTP4
full-vocabulary profile at a 180 W cap and a 200704-token context ceiling.

The control image is identified by image ID in `manifest.json`; installed worker
and Q128 binaries are hashed because repository source alone does not describe
the executing runtime. The experiment argument builder is side-effect free and
tested for the unchanged MTP4 defaults, omitted zero-token speculative config,
alternate quantization dry runs and invalid-value rejection.

The high-precision source lineage and draft-head storage identity remain marked
unresolved rather than inferred. No production file, image or service definition
was changed by M00.

Rollback: `systemctl --user start qwen38.service`.
