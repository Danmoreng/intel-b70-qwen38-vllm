"""Read-only thermal/frequency/cap checks, not a power comparison benchmark."""
import argparse,json,os,time
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--controller-pid',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    device=Path('/sys/bus/pci/devices/0000:03:00.0');freq=device/'tile0/gt0/freq0'
    hw=list((device/'hwmon').glob('hwmon*'));assert len(hw)==1
    paths=[hw[0]/'power1_cap',*hw[0].glob('temp*_input'),*(freq/x for x in ['act_freq','cur_freq','max_freq','min_freq'])]
    paths += list((freq/'throttle').glob('*'))
    assert not a.out.exists()
    with a.out.open('w') as f:
        while True:
            try:os.kill(a.controller_pid,0)
            except ProcessLookupError:break
            values={}
            for path in paths:
                if not path.is_file():continue
                try:values[str(path.relative_to(device))]=path.read_text().strip()
                except OSError:continue
            assert values[str((hw[0]/'power1_cap').relative_to(device))]=='180000000'
            f.write(json.dumps(dict(unix=time.time(),values=values,scope='Comparability only; observations before the first timestamp are unobserved'))+'\n');f.flush()
            time.sleep(15)
    print('COMPARABILITY_MONITOR_FINISHED',flush=True)
if __name__=='__main__':main()
