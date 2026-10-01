#!/usr/bin/env python3
"""Fresh-worker EXL3 diagnostics; leave GPTQ offline unless restoration is requested.

Instrumented runs are not claimed as unbiased performance benchmarks.
"""
import argparse
import base64
import concurrent.futures
import fcntl
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('migration_campaign_helpers',REPO/'scripts/run-web-coding-campaign.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
R=C.R
manifest_spec=importlib.util.spec_from_file_location('migration_read_only_manifest',REPO/'scripts/collect-exl3-migration-baseline.py')
MANIFEST=importlib.util.module_from_spec(manifest_spec);manifest_spec.loader.exec_module(MANIFEST)
NAME='b70-exl3-migration-contract';BASE='http://127.0.0.1:8082'
MODEL_DIR=Path.home()/'.cache/exl3xpu/turboderp-Qwen3.8-27B-exl3-4.00bpw'
SOURCE=REPO/'benchmarks/runs/2026-10-01-flappybird'
PROFILE='models/qwen3.8-27b-exl3-4.00bpw/migration-200704-c4.yaml'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def png_url(width=64,height=64):
    def chunk(kind,data): return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
    data+=chunk(b'IDAT',zlib.compress((b'\0'+b'\xff\0\0'*width)*height))+chunk(b'IEND',b'')
    return 'data:image/png;base64,'+base64.b64encode(data).decode()

def chat(text,**extra):
    return {'model':'Qwen3.8-27B','messages':[{'role':'user','content':text}],
            'temperature':0,'seed':20261001,'max_tokens':64,'reasoning_effort':'none',
            'chat_template_kwargs':{'enable_thinking':False},**extra}

def smoke(root,max_context=200704,media_limits=None):
    media_limits=media_limits or {'image':1,'video':0}
    image_limit=media_limits['image']
    records={}
    text=R.http(BASE,'/v1/chat/completions',chat('Name the capital of France. Answer with one word.'),timeout=180)
    assert 'paris' in text['choices'][0]['message']['content'].lower(),text
    records['text']=text
    R.save(root/'api-contract-progress.json',records)
    records['stream']=R.stream(BASE,chat('Count from one to ten.',stream=True,stream_options={'include_usage':True}),root/'stream.sse.jsonl')
    assert records['stream']['usage'] and records['stream']['message']['content']
    tool={'type':'function','function':{'name':'add','description':'Add two integers.',
          'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}}
    records['tool']=R.http(BASE,'/v1/chat/completions',chat('Call add with a=13 and b=29. Use the function; do not answer directly.',
                                                         tools=[tool],tool_choice='auto',max_tokens=128),timeout=180)
    R.save(root/'api-contract-progress.json',records)
    call=records['tool']['choices'][0]['message']['tool_calls'][0]['function']
    assert call['name']=='add' and json.loads(call['arguments'])=={'a':13,'b':29},call
    records['reasoning']=R.http(BASE,'/v1/chat/completions',chat('Calculate 13 times 29, then give the result.',
                                      reasoning_effort='low',chat_template_kwargs={'enable_thinking':True},max_tokens=256),timeout=180)
    message=records['reasoning']['choices'][0]['message']
    assert message.get('reasoning') or message.get('reasoning_content'),message
    assert '377' in (message.get('content') or ''),message
    image={'type':'image_url','image_url':{'url':png_url()}}
    image_payload=chat([image,{'type':'text','text':'What color is the square? Answer with one color word.'}])
    records['image']=R.http(BASE,'/v1/chat/completions',image_payload,timeout=180)
    R.save(root/'api-contract-progress.json',records)
    assert 'red' in records['image']['choices'][0]['message']['content'].lower(),records['image']
    if image_limit>1:
        records['images_at_limit']=R.http(BASE,'/v1/chat/completions',
            chat([image]*image_limit+[{'type':'text','text':'What color are all the squares? Answer with one word.'}]),timeout=600)
        assert 'red' in records['images_at_limit']['choices'][0]['message']['content'].lower()
    try:
        R.http(BASE,'/v1/chat/completions',chat([image]*(image_limit+1)+[{'type':'text','text':'Describe.'}]),timeout=30)
        raise AssertionError('Image request over configured count limit was accepted')
    except urllib.error.HTTPError as e:
        assert e.code==400,e.code
        records['over_image_limit']={'status':e.code,'error':e.read().decode()}
    records['prefix_first']=R.stream(BASE,chat('Prefix reuse check. '+('Stable independent sentence. '*1200),
                                     max_tokens=32,stream=True,stream_options={'include_usage':True}),root/'prefix-first.sse.jsonl')
    before=R.snapshot(BASE)
    records['prefix_extension']=R.stream(BASE,chat('Prefix reuse check. '+('Stable independent sentence. '*1200)+' Added final question: say OK.',
                                     max_tokens=32,stream=True,stream_options={'include_usage':True}),root/'prefix-extension.sse.jsonl')
    _,native=R.wait_accounted(BASE,before,records['prefix_extension']['usage'])
    assert native['cached_tokens']>0,native
    records['prefix_native']=native
    abort=chat('Write a very long numbered list of facts.',max_tokens=4096,ignore_eos=True,stream=True)
    request=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(abort).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=180) as response:
        for line in response:
            if line.startswith(b'data: ') and b'"content"' in line: break
    time.sleep(2)
    records['after_abort']=R.http(BASE,'/v1/chat/completions',chat('Reply OK.'),timeout=180)
    payloads=[chat(f'Independent concurrent request {i}. '+('Short deterministic note. '*500),
                  max_tokens=128,ignore_eos=True) for i in range(4)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        responses=list(pool.map(lambda p:R.http(BASE,'/v1/chat/completions',p,timeout=180),payloads))
    assert all(x['usage']['completion_tokens']==128 for x in responses)
    records['c4_usage']=[x['usage'] for x in responses]
    records['limits']=R.http(BASE,'/tokenize',{'model':'Qwen3.8-27B','prompt':'hello'})['max_model_len']
    assert records['limits']==max_context
    R.save(root/'api-contract.json',records)
    return {'status':'PASS','checks':list(records),'image_limit':image_limit,'video_limit_configured':media_limits['video'],
            'abort_note':'client disconnected after first streamed content; correlate trace with scheduler abort',
            'c4_note':'four simultaneous small requests; full long-context C4 load remains a separate release gate'}

def long_case(case,root,max_context=200704):
    if case=='near-limit':
        ids=R.http(BASE,'/tokenize',{'model':'Qwen3.8-27B','prompt':'function step(x) { return x + 1; }\nStable repeatable context.\n','add_special_tokens':False})['tokens']
        prompt_length=max_context-1024
        prompt=(ids*((prompt_length+len(ids)-1)//len(ids)))[:prompt_length]
        payload={'model':'Qwen3.8-27B','prompt':prompt,'temperature':0,'seed':20261001,
                 'max_tokens':1024,'ignore_eos':True}
        before=R.snapshot(BASE);start=time.monotonic()
        response=R.http(BASE,'/v1/completions',payload,timeout=1800)
        after,native=R.wait_accounted(BASE,before,response['usage'])
        assert response['usage']['prompt_tokens']==prompt_length
        result={'status':'COMPLETE','usage':response['usage'],'native':native,'native_before':before,
                'native_after':after,'wall_s':time.monotonic()-start,
                'prompt_ids_sha256':hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                'prompt_recipe':f'repeat frozen /tokenize code/prose token IDs to exactly {prompt_length}; output 1024; total {max_context}'}
    else:
        number={'103k':47,'139k':65,'188k':94}[case]
        source=SOURCE/f'cold-original-request-{number:04d}.json.gz'
        payload=json.loads(gzip.decompress(source.read_bytes()))
        payload.update(model='Qwen3.8-27B',max_tokens=1024,ignore_eos=True,
                       cache_salt='exl3-migration-old-fresh-'+case,stream=True,stream_options={'include_usage':True})
        preflight=R.bound_output(BASE,payload);assert payload['max_tokens']==1024
        before=R.snapshot(BASE);start=time.monotonic()
        response=R.stream(BASE,payload,root/'response.sse.jsonl')
        after,native=R.wait_accounted(BASE,before,response['usage'])
        assert response['usage']['prompt_tokens']==preflight['prompt_tokens']
        result={'status':'COMPLETE','source':str(source.relative_to(REPO)),'source_sha256':sha(source),
                'preflight':preflight,'usage':response['usage'],'native':native,
                'native_before':before,'native_after':after,'wall_s':time.monotonic()-start}
    assert result['usage']['completion_tokens']==1024
    assert native['cached_tokens']==0,native
    result['qualification']='instrumented capacity diagnostic; preemptions preserved for analysis, not silently accepted'
    R.save(root/'observation.json',result)
    return result

def regression(root):
    payload=chat('Facts to preserve: ALPHA=17, BETA=29.\n'+
                 ('Neutral padding paragraph. The landscape contains many ordinary stones and trees.\n'*1600)+
                 '\nReturn JSON with ALPHA, BETA, their sum, and a short JavaScript function that adds two integers.',
                 max_tokens=256,stream=True,stream_options={'include_usage':True},cache_salt='gdn-retirement-greedy-v1')
    response=R.stream(BASE,payload,root/'greedy.sse.jsonl')
    result={'status':'COMPLETE','response':response,
            'request_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
            'message_sha256':hashlib.sha256(json.dumps(response['message'],sort_keys=True).encode()).hexdigest(),
            'purpose':'Matched greedy output to detect semantic changes from GDN retirement; compare same frozen request across modes'}
    R.save(root/'greedy-regression.json',result)
    return result

def media_case(root,media_limits):
    """Real count-limit, video decoding, and oversized-image preprocessing probes."""
    video_limit=media_limits['video']
    if video_limit<1: raise ValueError('media-expanded requires video support in the candidate profile')
    clip=root/'red-2s.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-f','lavfi','-i',
                    'color=c=red:s=320x240:r=4','-t','2','-threads','1',
                    '-c:v','libx264','-pix_fmt','yuv420p',str(clip)],check=True)
    video={'type':'video_url','video_url':{'url':'data:video/mp4;base64,'+base64.b64encode(clip.read_bytes()).decode()}}
    question={'type':'text','text':'What color fills every video frame? Answer with one word.'}
    responses={}
    for count in sorted({1,video_limit}):
        responses[f'videos_{count}']=R.http(BASE,'/v1/chat/completions',chat([video]*count+[question]),timeout=600)
        assert 'red' in responses[f'videos_{count}']['choices'][0]['message']['content'].lower()
        R.save(root/'media-progress.json',responses)
    try:
        R.http(BASE,'/v1/chat/completions',chat([video]*(video_limit+1)+[question]),timeout=30)
        raise AssertionError('Video request over configured count limit was accepted')
    except urllib.error.HTTPError as e:
        assert e.code==400,e.code
        responses['over_video_limit']={'status':e.code,'error':e.read().decode()}
    large_image={'type':'image_url','image_url':{'url':png_url(3072,2048)}}
    responses['oversized_image']=R.http(BASE,'/v1/chat/completions',
        chat([large_image,{'type':'text','text':'What color is the image? Answer with one word.'}]),timeout=600)
    assert 'red' in responses['oversized_image']['choices'][0]['message']['content'].lower()
    # Qwen's image tokens represent 32x32 merged patches. An uncapped 3072x2048
    # image contributes 6144 visual tokens; the 4.2MP cap should stay near 4096.
    assert responses['oversized_image']['usage']['prompt_tokens']<4500,responses['oversized_image']['usage']
    result={'status':'PASS','responses':responses,'video_fixture_sha256':sha(clip),
            'fixture':'ffmpeg constant-red 320x240,4fps,2s; API decodes real H264 video',
            'scope':'Small-media count limits and one 6.3MP image; does not promise 32 maximum-area images plus 4 long videos at maximum text context'}
    R.save(root/'media-expanded.json',result)
    return result

def operational(case,root):
    if case in ('c4-long','c16-long'):
        concurrency,prompt_length=(4,32768) if case=='c4-long' else (16,8192)
        payloads=[]
        for i in range(concurrency):
            ids=R.http(BASE,'/tokenize',{'model':'Qwen3.8-27B','prompt':f'Request {i}: unique independent notes. function step(x) {{ return x+1; }}\n','add_special_tokens':False})['tokens']
            prompt=(ids*((prompt_length+len(ids)-1)//len(ids)))[:prompt_length]
            payloads.append({'model':'Qwen3.8-27B','prompt':prompt,'max_tokens':256,'ignore_eos':True,'temperature':0,'seed':20261001})
        before=R.snapshot(BASE)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            responses=list(pool.map(lambda p:R.http(BASE,'/v1/completions',p,timeout=1800),payloads))
        deadline=time.monotonic()+45
        while True:
            after=R.snapshot(BASE);native=R.delta(after,before)
            if native.get('completed')==concurrency and native.get('generation_tokens')==concurrency*256: break
            if time.monotonic()>deadline: raise RuntimeError('Concurrent accounting did not settle')
            time.sleep(.25)
        assert all(r['usage']['prompt_tokens']==prompt_length and r['usage']['completion_tokens']==256 for r in responses)
        assert native['preemptions']==0,native
        result={'status':'PASS','usage':[r['usage'] for r in responses],'native':native,
                'request_sha256':[hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest() for p in payloads],
                'purpose':f'{concurrency} simultaneous independent {prompt_length}-token prompts; fixed output 256 each; not a claim of {concurrency} simultaneous maximum-context sequences'}
    elif case=='image-long':
        content=[{'type':'image_url','image_url':{'url':png_url()}},
                 {'type':'text','text':('Neutral independent context note. function step(x) { return x+1; }\n'*7000)+
                  '\nWhat color is the square at the beginning? Answer with one color word.'}]
        before=R.snapshot(BASE)
        response=R.http(BASE,'/v1/chat/completions',chat(content),timeout=1800)
        _,native=R.wait_accounted(BASE,before,response['usage'])
        assert response['usage']['prompt_tokens']>100000,response['usage']
        assert 'red' in response['choices'][0]['message']['content'].lower(),response
        assert native['preemptions']==0,native
        result={'status':'PASS','response':response,'native':native,'purpose':'One allowed image in a long context'}
    else:
        prefix='ALPHA=17 and BETA=29.\n'+('Neutral reproducible context note with ordinary facts.\n'*5000)
        payload=chat(prefix+'\nReturn ALPHA and BETA as JSON.',max_tokens=256,stream=True,
                     stream_options={'include_usage':True},cache_salt='long-extension-abort-v1')
        first=R.stream(BASE,payload,root/'first.sse.jsonl')
        assert first['usage']['prompt_tokens']>32000
        # Extend the same history; retained-prefix behavior must be observable.
        extension=chat(prefix+'\nReturn ALPHA and BETA as JSON. Also give their sum.',max_tokens=256,stream=True,
                       stream_options={'include_usage':True},cache_salt='long-extension-abort-v1')
        before=R.snapshot(BASE)
        extended=R.stream(BASE,extension,root/'extension.sse.jsonl')
        _,native=R.wait_accounted(BASE,before,extended['usage'])
        assert native['cached_tokens']>0 and native['preemptions']==0,native
        abort=chat(prefix+'\nWrite a long numbered list of facts.',max_tokens=4096,ignore_eos=True,stream=True,
                   cache_salt='long-extension-abort-v1')
        req=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(abort).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=1800) as response:
            for line in response:
                if line.startswith(b'data: ') and b'"content"' in line: break
        time.sleep(2)
        restarted=R.http(BASE,'/v1/chat/completions',chat(prefix+'\nReturn ALPHA, BETA and their sum as JSON.',max_tokens=256,
                                                       cache_salt='long-extension-abort-v1'),timeout=1800)
        assert '17' in restarted['choices'][0]['message']['content'] and '29' in restarted['choices'][0]['message']['content']
        result={'status':'PASS','first_usage':first['usage'],'extension_usage':extended['usage'],
                'extension_native':native,'restart':restarted,'purpose':'Long prefix extension and client abort/restart; trace must confirm FINISHED_ABORTED'}
    R.save(root/'operational.json',result)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image',default='local/exl3xpu:migration-old-safe-20261001')
    ap.add_argument('--runtime',choices=('old','target'),default='old',help='Pinned ABI/profile/cache namespace')
    ap.add_argument('--settings-json',type=Path,help='Explicit vLLM setting overrides; base image profile remains immutable')
    ap.add_argument('--cases',default='smoke')
    ap.add_argument('--retirement-fix',choices=('0','1'),help='Explicit old-allocator A/B on one immutable candidate image')
    ap.add_argument('--restore-production',action='store_true',help='Explicit pinned GPTQ rollback drill after the campaign')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();cases=a.cases.split(',')
    settings=json.loads(a.settings_json.read_text()) if a.settings_json else {}
    if not isinstance(settings,dict): raise SystemExit('Settings must be a JSON object')
    max_context=settings.get('max_model_len',200704)
    media_limits=settings.get('limit_mm_per_prompt',{'image':1,'video':0})
    if any(not isinstance(v,int) or v<0 for v in media_limits.values()): raise SystemExit('Invalid media count limits')
    profile=(PROFILE if a.runtime=='old' else 'models/qwen3.8-27b-exl3-4.00bpw/migration-target-200704-c4.yaml')
    if any(c not in ('smoke','regression','103k','139k','188k','near-limit','c4-long','c16-long','image-long','extension-abort-long','media-expanded') for c in cases): raise SystemExit('Unknown case')
    if a.output.exists(): raise SystemExit('Fresh output directory required')
    release=json.loads((REPO/'config/production_image.json').read_text())
    image=subprocess.check_output(['docker','image','inspect',a.image,'--format','{{.Id}}'],text=True).strip()
    if image==release['image_id']: raise SystemExit('Candidate must be separate from GPTQ production')
    lockpath=REPO.parent/'Local-AI-B70/qwen38/context-benchmark/run.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        a.output.mkdir(parents=True);state={'schema':1,'started_at':R.now(),'image':image,'cases':{},'status':'RUNNING',
                                          'purpose':a.runtime+'-ABI instrumented contract diagnostics','settings_overrides':settings,
                                          'settings_sha256':sha(a.settings_json) if a.settings_json else None,'production_restored':False,
                                          'restore_production_requested':a.restore_production}
        def save(): R.save(a.output/'campaign.json',state)
        def interrupted(*args): raise RuntimeError('Migration diagnostic interrupted')
        signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
        save()
        try:
            C.command(['systemctl','--user','stop',C.SERVICE])
            for case in cases:
                root=(a.output/case).resolve();root.mkdir();(root/'trace').mkdir()
                cache=Path.home()/('.cache/exl3xpu/migration-'+a.runtime)/image.removeprefix('sha256:')
                for d in ('vllm','triton','neo_compiler_cache'): (cache/d).mkdir(parents=True,exist_ok=True)
                launch=['docker','run','-d','--name',NAME,'--device','/dev/dri',
                        '-v','/dev/dri/by-path:/dev/dri/by-path:ro','--shm-size','8g',
                        '-p','127.0.0.1:8082:8000','-e','HF_HUB_OFFLINE=1',
                        '-e','PYTHONPATH=/opt/b70-runtime','-e','EXL3_MIGRATION_TRACE=/results/trace',
                        '-e','EXL3_LOADER_REPORT_DIR=/results','-v',str(root)+':/results',
                        '-v',str(MODEL_DIR)+':/models/checkpoint:ro',
                        '-v',str(REPO/'runtime')+':/opt/b70-runtime:ro',
                        '-v',str(cache/'vllm')+':/root/.cache/vllm',
                        '-v',str(cache/'triton')+':/root/.triton/cache',
                        '-v',str(cache/'neo_compiler_cache')+':/root/.cache/neo_compiler_cache',
                        image,profile,'--gpu','0','--port','8000','--model-path','/models/checkpoint']
                for key,value in settings.items():
                    launch+=['--set','vllm.'+key+'='+json.dumps(value,separators=(',',':'))]
                if a.retirement_fix is not None:
                    image_position=launch.index(image)
                    launch[image_position:image_position]=['-e','EXL3_FIX_SPARSE_GDN_RETIREMENT='+a.retirement_fix]
                state['phase']=case+'-startup';state['cases'][case]={'launch':launch};save()
                print('START '+case,flush=True)
                try:
                    C.command(launch)
                    deadline=time.monotonic()+900
                    while True:
                        try:
                            R.http(BASE,'/v1/models',timeout=3)
                            identity=R.identity(NAME)
                            assert identity['image_id']==image
                            break
                        except (OSError,subprocess.CalledProcessError):
                            inspect_result=subprocess.run(['docker','inspect',NAME],capture_output=True,text=True)
                            if inspect_result.returncode==0 and not json.loads(inspect_result.stdout)[0]['State']['Running']:
                                raise RuntimeError('Candidate exited during startup; inspect worker.log')
                            if time.monotonic()>deadline: raise RuntimeError('Candidate startup exceeded 900 seconds')
                            time.sleep(2)
                    R.cap();state['cases'][case]['identity']=identity;save()
                    state['cases'][case]['result']=(smoke(root,max_context,media_limits) if case=='smoke' else regression(root) if case=='regression'
                                                    else media_case(root,media_limits) if case=='media-expanded'
                                                    else operational(case,root) if case in ('c4-long','c16-long','image-long','extension-abort-long')
                                                    else long_case(case,root,max_context))
                    runtime=MANIFEST.probe(container=NAME)
                    R.save(root/'runtime-environment.json',runtime)
                    state['cases'][case]['runtime_manifest_sha256']=sha(root/'runtime-environment.json')
                    print('COMPLETE '+case,flush=True);save()
                finally:
                    logs=subprocess.run(['docker','logs',NAME],capture_output=True,text=True)
                    (root/'worker.log').write_text(logs.stdout+logs.stderr)
                    subprocess.run(['docker','stop','-t','20',NAME],capture_output=True,timeout=60)
                    subprocess.run(['docker','rm','-f',NAME],capture_output=True,timeout=30)
            state['status']='COMPLETE'
        except BaseException as e:
            state['status']='FAILED';state['error']=repr(e);save();raise
        finally:
            subprocess.run(['docker','rm','-f',NAME],capture_output=True,timeout=30)
            if a.restore_production:
                C.command(['systemctl','--user','start',C.SERVICE])
                restored=C.wait_health('http://127.0.0.1:8081','b70-qwen38-vllm',release['image_id'])
                state['production_restored']=restored['image_id']==release['image_id']
                state['restored_identity']=restored
            else:
                C.command(['systemctl','--user','stop',C.SERVICE])
                state['production_left_offline']=True
            state['finished_at']=R.now();save()
            print(json.dumps({'status':state['status'],'production_restored':state['production_restored']}),flush=True)

if __name__=='__main__': main()
