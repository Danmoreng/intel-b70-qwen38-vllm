"""Launch a pinned EXL3 profile as an inspectable direct vLLM process."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
PROFILE = 'models/qwen3.8-27b-exl3-4.00bpw/migration-target-optimized.yaml'
MODEL = Path.home()/'.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw'
BASE = 'http://127.0.0.1:8082'


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def profile(image, overrides=None):
    source = """import json,sys; sys.path.insert(0,'scripts'); import serve
c=serve.load(sys.argv[1]); print(json.dumps({'config':c,'argv':serve.vllm_argv(c,'/models/checkpoint',8000,[]),'env':{k:str(v).replace('{model_dir}',c['_dir']) for k,v in c['env'].items() if v is not None}}))"""
    d=json.loads(subprocess.check_output(['docker','run','--rm','--entrypoint','python',image,
                                        '-c',source,PROFILE],text=True))
    d['env'].update(overrides or {})
    return d


class Worker:
    def __init__(self,image,root,name='b70-exl3-qualification',env=None):
        self.image=subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()
        self.root=Path(root).resolve(); self.name=name; self.profile=profile(self.image,env)

    def start(self):
        if subprocess.run(['docker','inspect',self.name],capture_output=True).returncode==0:
            raise RuntimeError('Refuse to replace existing worker '+self.name)
        self.root.mkdir(parents=True,exist_ok=True)
        env={**self.profile['env'],'HF_HUB_OFFLINE':'1','PYTHONPATH':'/opt/b70-runtime',
             'ZE_AFFINITY_MASK':'0','EXL3_LOADER_REPORT_DIR':'/results/loader'}
        command=['docker','run','-d','--name',self.name,'--device','/dev/dri','--shm-size','8g',
                 '-p','127.0.0.1:8082:8000','-v',str(MODEL)+':/models/checkpoint:ro',
                 '-v',str(REPO/'runtime')+':/opt/b70-runtime:ro','-v',str(self.root)+':/results']
        for k,v in env.items(): command += ['-e',k+'='+v]
        cache=Path.home()/'.cache/exl3xpu/optimized-qualification'/self.image.removeprefix('sha256:')
        for k,target in [('vllm','/root/.cache/vllm'),('triton','/root/.triton/cache'),('neo_compiler_cache','/root/.cache/neo_compiler_cache')]:
            (cache/k).mkdir(parents=True,exist_ok=True);command+=['-v',str(cache/k)+':'+target]
        command+=['--entrypoint','/opt/venv/bin/vllm',self.image,*self.profile['argv'][1:]]
        self.command=command
        (self.root/'launch.json').write_text(json.dumps({'image_id':self.image,'profile':self.profile,'command':command},indent=2)+'\n')
        subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
        deadline=time.monotonic()+900
        while True:
            try:
                with urllib.request.urlopen(BASE+'/v1/models',timeout=3) as r: json.load(r)
                break
            except OSError:
                item=json.loads(subprocess.check_output(['docker','inspect',self.name],text=True))[0]
                if not item['State']['Running']:raise RuntimeError('Candidate worker exited during startup')
                if time.monotonic()>deadline:raise TimeoutError('Candidate worker startup')
                time.sleep(2)
        item=json.loads(subprocess.check_output(['docker','inspect',self.name],text=True))[0]
        assert item['Image']==self.image
        (self.root/'identity.json').write_text(json.dumps(item,indent=2)+'\n')
        return self

    def stop(self):
        logs=subprocess.run(['docker','logs',self.name],capture_output=True,text=True)
        (self.root/'worker.log').write_text(logs.stdout+logs.stderr)
        subprocess.run(['docker','rm','-f',self.name],capture_output=True,timeout=60)
