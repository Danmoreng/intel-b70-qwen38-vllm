#!/usr/bin/env python3
"""Wait for the successful owned power campaign, publish measurements and FF main."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--unit',required=True)
    a=p.parse_args()
    while True:
        value=subprocess.check_output(['systemctl','--user','show',a.unit,'--property=ActiveState','--value'],text=True).strip()
        if value not in ('active','activating','deactivating'):break
        time.sleep(10)
    status=subprocess.check_output(['systemctl','--user','show',a.unit,'--property=ExecMainStatus','--value'],text=True).strip()
    result=subprocess.check_output(['systemctl','--user','show',a.unit,'--property=Result','--value'],text=True).strip()
    assert status=='0' and result=='success', 'Measurement unit failed; preserve evidence, do not publish'
    state=json.loads((a.campaign/'campaign.json').read_text())
    assert state['status']=='COMPLETE_POWER_COMPARISON_MEASUREMENTS' and state['success']
    assert state['restored_power_cap_w']==180
    assert subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()=='', 'Working tree changed; do not commit unrelated work'
    cap=next(Path('/sys/bus/pci/devices/0000:03:00.0/hwmon').glob('*/power1_cap'))
    assert int(cap.read_text())==180000000
    live=json.loads(subprocess.check_output(['docker','inspect','b70-qwen38-vllm'],text=True))[0]
    assert live['Image']==state['image_id'] and live['State']['Running']
    with urllib.request.urlopen('http://127.0.0.1:8081/v1/models',timeout=10) as response:
        assert any(row['id']=='Qwen3.8-27B' for row in json.load(response)['data'])
    subprocess.run(['/usr/bin/python3',str(REPO/'scripts/publish-exl3-power-comparison.py'),
        '--campaign',str(a.campaign),'--baseline',str(a.baseline),'--out',str(a.out)],check=True,cwd=REPO)
    subprocess.run(['/usr/bin/python3','-m','unittest','discover','-s','tests/unit','-v'],check=True,cwd=REPO,
        env={**__import__('os').environ,'PYTHONPATH':str(REPO/'scripts')+':'+str(REPO)})
    subprocess.run(['git','diff','--check'],check=True,cwd=REPO)
    subprocess.run(['git','add','README.md','docs/EXL3_POWER_COMPARISON.md',str(a.out.relative_to(REPO))],check=True,cwd=REPO)
    subprocess.run(['git','diff','--cached','--check'],check=True,cwd=REPO)
    subprocess.run(['git','commit','-m','Document full EXL3 v2 benchmarks at 180 W, 230 W and 275 W'],check=True,cwd=REPO)
    subprocess.run(['git','fetch','origin','main'],check=True,cwd=REPO)
    subprocess.run(['git','merge-base','--is-ancestor','origin/main','HEAD'],check=True,cwd=REPO)
    subprocess.run(['git','push','origin','HEAD:main'],check=True,cwd=REPO)
    receipt=dict(status='PASS_POWER_COMPARISON_PUBLISHED',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        image_id=state['image_id'],restored_power_w=180,finished_unix=time.time())
    (a.campaign/'publication.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
