"""Diagnostic-only real activation capture for M05/M06 offline gates."""
from pathlib import Path
import threading

import torch


ROOT = Path("/evidence")
LIMIT = 64
LOCK = threading.Lock()
ROWS = {"draft": [], "target": []}
DONE = set()
SEEN = set()


def capture(kind: str, hidden_states: torch.Tensor) -> None:
    if kind not in SEEN:
        SEEN.add(kind)
        print(f"B70_CAPTURE_HOOK {kind} {tuple(hidden_states.shape)} gate={(ROOT / 'capture').exists()}", flush=True)
    if kind in DONE or not (ROOT / "capture").exists():
        return
    value = hidden_states.detach().reshape(-1, hidden_states.shape[-1]).cpu()
    with LOCK:
        ROWS[kind].append(value)
        total = sum(part.shape[0] for part in ROWS[kind])
        if total >= LIMIT or (ROOT / "done").exists():
            data = torch.unique(torch.cat(ROWS[kind], dim=0), dim=0)[:LIMIT].contiguous()
            torch.save({"hidden_states": data, "kind": kind}, ROOT / f"{kind}.pt")
            DONE.add(kind)
            ROWS[kind].clear()
            print(f"B70_CAPTURE_HIDDEN {kind} {tuple(data.shape)} {data.dtype}", flush=True)
