# Qualified EXL3 v1 rollback

The immutable `production-exl3-v1` tag, policy, original receipts, runtime source and compiler namespace remain available. The strict rollback wrapper preserves the local host/port settings and selects these saved release files independently of the current profile.

```bash
scripts/run-server-exl3-v1-rollback.sh --check-only
systemctl --user stop b70-qwen38-vllm.service
scripts/run-server-exl3-v1-rollback.sh
```

The last command serves v1 in the foreground. Confirm `/v1/models`, a real response and the saved image ID. Stop it before starting the current service. For a persistent rollback, restore the three saved policy/image files into `config/`, set `VLLM_IMAGE=local/b70-qwen38-vllm:production-exl3-v1` in the local `.env`, then start the user service. Keep the saved receipts and v1 tag intact. These are user-service operations; no sudo or driver installation is required.
