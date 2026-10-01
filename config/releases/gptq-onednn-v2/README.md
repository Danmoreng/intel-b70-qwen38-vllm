# Pinned GPTQ rollback

Original policy4ce5f3bd and image ed1ebca7 are preserved. The rollback launcher reads this frozen configuration and environment defaults, independent of future EXL3 canonical files. Runtime caches remain keyed by this policy and image; EXL3 uses a different directory and also isolates neo compiler caches. A local `.env` may preserve host/port settings; it is not committed.

Stop `b70-qwen38-vllm.service` before starting `scripts/run-server-gptq-rollback.sh`. Both use the same container name and API endpoint, so they cannot run simultaneously. To make rollback persistent, point the user service ExecStart at this launcher, reload the unit and restart. Check `/v1/models`, the immutable image ID and a real chat/tool request. Do not rebuild/reuse EXL3 binary or compiled caches with GPTQ. The final release report records the actual rollback exercise.
