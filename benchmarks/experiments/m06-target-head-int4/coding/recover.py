#!/usr/bin/env python3
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);run=p.parse_args().run_dir
if (run/'production-owned').exists():
    for file in run.glob('arm-*/sandbox-scope.json'):
        value=json.loads(file.read_text())['runId']
        if not value or any(c not in '0123456789abcdef' for c in value):raise RuntimeError('invalid scope')
        subprocess.run(['systemctl','--user','kill','--kill-whom=all','--signal=SIGTERM',f'local-ai-b70-job-{value}.scope'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
    subprocess.run(['python3','/home/sebastian/LocalLLM/intel-b70-qwen38-vllm/scripts/restore-production.py'],check=True,timeout=850,env={**os.environ,'B70_RUN_DIR':str(run)})
    (run/'production-owned').unlink()
