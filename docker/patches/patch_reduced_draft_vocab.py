#!/usr/bin/env python3
"""Add an optional reduced-token mapping to the existing B70 INT4 draft head."""

from __future__ import annotations

import os
import sys


MARKER = "B70_DRAFT_VOCAB_PATH"


def replace_once(text: str, old: str, new: str, description: str) -> str:
    if old not in text:
        sys.exit(f"anchor not found while patching {description}")
    return text.replace(old, new, 1)


def main() -> None:
    import vllm

    helper = os.path.join(
        os.path.dirname(vllm.__file__),
        "model_executor",
        "models",
        "b70_draft_lmhead_int4.py",
    )
    text = open(helper).read()
    if MARKER in text:
        print(f"already patched {helper}")
        return

    text = replace_once(
        text,
        "import os\n\nimport torch\n",
        "import os\nfrom pathlib import Path\n\nimport torch\n",
        "Path import",
    )
    text = replace_once(
        text,
        '''\
    print("[B70] draft LM head INT4: cuantizando lm_head fp16 "
          f"{tuple(weight.shape)} -> INT4 g128 sym (one-time)", flush=True)
    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(
        weight.detach()
    )
    model._b70_lmhead_int4 = (qweight, scales, qzeros, group_size)
    fp16_bytes = weight.numel() * weight.element_size()
''',
        '''\
    vocab_path = os.environ.get("B70_DRAFT_VOCAB_PATH")
    token_ids = None
    quant_weight = weight.detach()
    if vocab_path:
        try:
            ids = [
                int(line)
                for line in Path(vocab_path).read_text().splitlines()
                if line.strip()
            ]
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"invalid B70 draft vocabulary {vocab_path}: {exc}"
            ) from exc
        if not ids or len(ids) != len(set(ids)):
            raise RuntimeError("B70 draft vocabulary must be non-empty and unique")
        if min(ids) < 0 or max(ids) >= weight.shape[0]:
            raise RuntimeError(
                "B70 draft vocabulary contains an out-of-range token ID"
            )
        if len(ids) % 8:
            raise RuntimeError("B70 draft vocabulary size must be divisible by 8")
        token_ids = torch.tensor(ids, dtype=torch.long, device=weight.device)
        quant_weight = weight.detach().index_select(0, token_ids)
        print(
            f"[B70] reduced draft vocabulary: {len(ids)}/{weight.shape[0]} "
            f"tokens from {vocab_path}",
            flush=True,
        )
    print("[B70] draft LM head INT4: cuantizando lm_head fp16 "
          f"{tuple(quant_weight.shape)} -> INT4 g128 sym (one-time)", flush=True)
    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(quant_weight)
    model._b70_lmhead_int4 = (
        qweight,
        scales,
        qzeros,
        group_size,
        token_ids,
        weight.shape[0],
    )
    fp16_bytes = weight.numel() * weight.element_size()
''',
        "reduced weight construction",
    )
    text = replace_once(
        text,
        '''\
    qweight, scales, qzeros, group_size = model._b70_lmhead_int4
    logits = int4_lmhead_logits(
        hidden_states, qweight, scales, qzeros, group_size
    )
''',
        '''\
    qweight, scales, qzeros, group_size, token_ids, full_vocab_size = (
        model._b70_lmhead_int4
    )
    logits = int4_lmhead_logits(
        hidden_states, qweight, scales, qzeros, group_size
    )
    if token_ids is not None:
        reduced_logits = logits
        logits = torch.full(
            (*reduced_logits.shape[:-1], full_vocab_size),
            -torch.inf,
            dtype=reduced_logits.dtype,
            device=reduced_logits.device,
        )
        logits.index_copy_(-1, token_ids, reduced_logits)
''',
        "logit remapping",
    )

    compile(text, helper, "exec")
    open(helper, "w").write(text)
    print(f"patched {helper}")


if __name__ == "__main__":
    main()
