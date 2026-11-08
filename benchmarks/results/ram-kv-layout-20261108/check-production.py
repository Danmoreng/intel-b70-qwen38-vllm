#!/usr/bin/env python3
"""Integration probes for the 2026-11-08 RAM-layout correction; no service mutations."""
import argparse,base64,hashlib,importlib.util,json,struct,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser(description='Tool/vision and optional natural RAM restoration checks on an idle production API.')
p.add_argument('--base',default='http://127.0.0.1:8081')
p.add_argument('--out',required=True,type=Path)
p.add_argument('--natural-restore',action='store_true')
a=p.parse_args()
OUT=a.out
OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('cache_checks',ROOT/'scripts/test-kv-cache-candidate.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
BASE=a.base
def chat(body):
 body={'model':'Qwen3.8-27B','temperature':0,'max_tokens':256,'chat_template_kwargs':{'enable_thinking':False},**body}
 return b.post(BASE,'/v1/chat/completions',body,timeout=180)
result={}
tool=chat({'messages':[{'role':'user','content':'Use the get_weather tool to get the weather for Berlin. Call the tool now.'}], 'tools':[{'type':'function','function':{'name':'get_weather','description':'Get current weather for a city','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}], 'tool_choice':'auto'})
calls=tool['choices'][0]['message'].get('tool_calls',[])
assert len(calls)==1 and calls[0]['function']['name']=='get_weather',tool
assert json.loads(calls[0]['function']['arguments'])['city'].lower()=='berlin',tool
result['tool']={'passed':True,'message':tool['choices'][0]['message'],'usage':tool['usage']}
def chunk(name,data):return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',64,64,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+bytes((255,0,0))*64)*64))+chunk(b'IEND',b'')
vision=chat({'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}},{'type':'text','text':'What is the solid color in this image? Reply with one English color word.'}]}]})
assert vision['choices'][0]['message']['content'].strip().lower().strip('.')=='red',vision
result['vision']={'passed':True,'content':vision['choices'][0]['message']['content'],'usage':vision['usage'],'fixture':'64x64 RGB solid red PNG','fixture_sha256':hashlib.sha256(png).hexdigest()}
(OUT/'smokes.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)

if a.natural_restore:
    import uuid
    key='retention-layout-'+uuid.uuid4().hex[:8]
    favored=b.prompt(BASE,48000,key)
    large=b.prompt(BASE,261900,'pressure-layout-'+uuid.uuid4().hex[:8])
    result={'power_cap_w':180,'method':'Natural GPU prefix eviction by a near-262K competing prompt; production developer mode remains disabled. Reuse the favored history three times to exercise ARC.','rows':[]}
    def record(label,messages):
     row=b.request(BASE,messages);result['rows'].append(dict(label=label,**row));(OUT/'natural-restore.json').write_text(json.dumps(result,indent=2)+'\n');return row
    before=b.metrics(BASE)
    assert before['vllm:num_requests_running']==before['vllm:num_requests_waiting']==0
    for i in range(3):
     row=record('favored-'+str(i),favored)
     assert all(code in row['output'] for code in b.needles(key)),row['output']
    record('near-max-context-pressure',large)
    resume_before=b.metrics(BASE)
    row=record('restored-after-pressure',favored)
    after=b.metrics(BASE)
    result['restore_delta']={k:v-resume_before.get(k,0) for k,v in after.items() if any(s in k for s in ['prefix_cache','kv_offload'])}
    result['preemptions']=after['vllm:num_preemptions_total']-before['vllm:num_preemptions_total']
    result['correct_markers']=all(code in row['output'] for code in b.needles(key))
    result['same_answer']=row['outputSha256']==result['rows'][0]['outputSha256']
    result['passed']=result['correct_markers'] and result['same_answer'] and result['preemptions']==0 and result['restore_delta'].get('vllm:external_prefix_cache_hits_total',0)>0 and result['restore_delta'].get('vllm:kv_offload_total_bytes_total:CPU_to_GPU',0)>0
    (OUT/'natural-restore.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    if not result['passed']:raise RuntimeError('Natural restoration check failed; see saved receipt')
