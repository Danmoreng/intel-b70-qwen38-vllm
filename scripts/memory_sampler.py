#!/usr/bin/env python3
"""Sample per-DRM-client VRAM accounting from inside the engine container."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/evidence/memory-samples.jsonl"))
    parser.add_argument("--stop-file", type=Path, default=Path("/evidence/stop-memory-sampler"))
    parser.add_argument("--pci", default="0000:03:00.0")
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args()
    if not 0.05 <= args.interval <= 10:
        parser.error("--interval must be between 0.05 and 10 seconds")

    with args.output.open("x") as output:
        while not args.stop_file.exists():
            clients = {}
            for path in Path("/proc").glob("[0-9]*/fdinfo/*"):
                try:
                    text = path.read_text()
                except OSError:
                    continue
                if args.pci not in text:
                    continue
                fields = dict(re.findall(r"^(drm-[^:]+):\s*(.+)$", text, re.MULTILINE))
                client_id = fields.get("drm-client-id")
                if client_id:
                    clients[client_id] = {"pid": int(path.parts[2]), **fields}
            output.write(json.dumps({"time": time.time(), "clients": clients}) + "\n")
            output.flush()
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
