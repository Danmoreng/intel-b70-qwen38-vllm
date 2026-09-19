"""Read-only host telemetry; no driver context, model requests or GPU kernels."""
import json
from pathlib import Path
import sys
import time

run=Path(sys.argv[1])
device=Path('/sys/bus/pci/devices/0000:03:00.0')
hw=next(device.glob('hwmon/hwmon*'))
freqs=list(device.glob('tile*/gt*/freq*/*freq'))
deadline=time.monotonic()+30
while not (run/'production-owned').exists() and time.monotonic()<deadline:
    time.sleep(.1)
with (run/'hardware.jsonl').open('a') as out:
    while (run/'production-owned').exists():
        row={'time_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),
             'sensors':{p.name:int(p.read_text()) for p in hw.iterdir()
                        if p.name in ('energy1_input','power1_cap') or
                        (p.name.startswith('temp') and p.name.endswith('_input'))},
             'frequency':{str(p.relative_to(device)):p.read_text().strip() for p in freqs}}
        try:row['phase']=json.loads((run/'progress.json').read_text())
        except (OSError,ValueError):pass
        out.write(json.dumps(row)+'\n');out.flush()
        time.sleep(.5)
