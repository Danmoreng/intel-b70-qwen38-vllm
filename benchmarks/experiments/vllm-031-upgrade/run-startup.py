#!/usr/bin/env python3
"""First live startup and bounded API/MTP probe of the frozen new image."""
import fcntl,json,sys,time,urllib.request
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO/'scripts'))
from exl3_candidate_worker import Worker,BASE
r=json.loads((REPO/'config/experiments/vllm-031-upgrade/production_image.json').read_text())
root=REPO/'benchmark-results/vllm-031-upgrade-20261009/startup-attempt1'
w=Worker(r['image_id'],root,'b70-v031-startup')
with (REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  w.start();print('READY',flush=True)
  payload=dict(model='Qwen3.8-27B',messages=[dict(role='user',content='Write a short Python function that adds two integers.')],temperature=0,max_tokens=128,chat_template_kwargs={'enable_thinking':False})
  request=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=180) as response:d=json.load(response)
  (root/'reply.json').write_text(json.dumps(d,indent=2)+'\n')
  assert 'def ' in d['choices'][0]['message']['content']
  (root/'result.json').write_text(json.dumps(dict(status='PASS_FIRST_NEW_IMAGE_STARTUP',image_id=w.image,usage=d['usage']),indent=2)+'\n');print('PASS_STARTUP',flush=True)
 finally:w.stop()
