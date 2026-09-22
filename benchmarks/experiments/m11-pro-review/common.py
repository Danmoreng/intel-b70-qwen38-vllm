"""Shared exclusive GPU lifecycle; each experiment records its own frozen base."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
diag=load('m11_diag',REPO/'scripts/run-diagnostics.py')
BASE='sha256:b675d81d4e7cc63fbcd6df395965ea16ec5c4704428c81118a1618185245dd5a'
NAME='b70-pro-review-arm'
OPERATOR='b70-pro-review-replay'

def save(p,data):p.write_text(json.dumps(data,indent=2)+'\n')

def recover(run):
    if not (run/'production-owned').exists():return
    for name in (NAME,OPERATOR):
        subprocess.run(['docker','stop','-t','20',name],capture_output=True,timeout=40)
    subprocess.run(['python3',str(REPO/'scripts/restore-production.py')],check=True,timeout=850,
                   env={**os.environ,'B70_RUN_DIR':str(run)})
    (run/'production-owned').unlink()

class Session:
    def __init__(self,run):self.run=run;self.process=None;self.log=None
    def __enter__(self):
        self.run.mkdir(parents=True,exist_ok=True)
        self.lock=Path('/home/sebastian/LocalLLM/Local-AI-B70/qwen38/context-benchmark/run.lock').open('a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.production=diag.production_inspect()
        assert self.production['Image']==BASE
        diag.ensure_idle(diag.PRODUCTION_URL)
        assert int(diag.POWER_CAP.read_text())==180000000
        save(self.run/'production-reference.json',{'image':BASE,'args':self.production['Args'],
             'environment':[x for x in self.production['Config']['Env'] if x.startswith(('B70_','VLLM_','ZE_','PYTORCH_'))]})
        (self.run/'production-owned').touch()
        subprocess.run(['systemctl','--user','stop','qwen38.service'],check=True,timeout=120)
        return self
    def start(self,label,image,*,flags=None,extra=None,budget=4096,cache_key=None):
        self.stop()
        out=self.run/label;out.mkdir(parents=True,exist_ok=True)
        args=self.production['Args'].copy()
        args[args.index('--max-num-batched-tokens')+1]=str(budget)
        args+=extra or []
        cmd=diag.engine_command(self.production,name=NAME,image=image,evidence=out,arguments=args)
        # Cache separation is per actual arm, retained across repeated starts.
        for i,a in enumerate(cmd):
            if a.endswith(':/root/.cache/vllm') or a.endswith(':/root/.triton/cache'):
                dest=a.split(':',1)[1];p=self.run/'compiler-cache'/str(cache_key if cache_key is not None else (flags or {}).get('B70_FUSED_QK_ROPE_GATE','0'))/str(budget)/('vllm' if 'vllm' in dest else 'triton')
                p.mkdir(parents=True,exist_ok=True);cmd[i]=str(p)+':'+dest
        at=cmd.index(image)
        env={'VLLM_SERVER_DEV_MODE':'1','PYTHONUNBUFFERED':'1',**(flags or {})}
        cmd[at:at]=[v for k,val in env.items() for v in ('-e',k+'='+str(val))]
        save(out/'command.json',cmd)
        self.log=(out/'server.log').open('w')
        self.process=subprocess.Popen(cmd,stdout=self.log,stderr=subprocess.STDOUT)
        diag.wait_ready(self.process,timeout=900)
        actual=json.loads(subprocess.check_output(['docker','inspect',NAME]))[0]
        save(out/'runtime.json',{'image':actual['Image'],'args':actual['Args'],'environment':env})
        assert diag.api('/v1/models')['data'][0]['max_model_len']==200704
        return out
    def stop(self):
        if self.process:
            subprocess.run(['docker','stop','-t','30',NAME],capture_output=True,timeout=50)
            self.process.wait(timeout=30);self.process=None
        if self.log:self.log.close();self.log=None
    def __exit__(self,*exc):
        try:self.stop()
        finally:recover(self.run)

def reset():
    result=diag.api('/reset_prefix_cache',{})
    assert result=={'success':True},result
