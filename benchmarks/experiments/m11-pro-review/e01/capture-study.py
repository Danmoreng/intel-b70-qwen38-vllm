import base64
import json
from pathlib import Path
import signal
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import Session,recover,diag,save,reset,ROOT,REPO,OPERATOR

run=Path(sys.argv[-1])
if sys.argv[1]=='recover':recover(run);sys.exit()
signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
screen=Path((ROOT/'e01/LATEST-SCREEN').read_text().strip())
assert json.loads((screen/'screen-decision.json').read_text())['pass']
with Session(run) as session:
    out=session.start('capture','local/qwen38-b70-vllm:e01-capture-20260919',
         flags={'B70_FUSED_QK_ROPE_GATE':'0'},extra=['--enforce-eager','--compilation-config','{"mode":0}'])
    data=out/'activations';data.mkdir()
    messages=[{'role':'user','content':'Review this Python cache implementation and explain thread safety, eviction and correctness tests.\n'+
        ('def cache_get(key, cache):\n    return cache.get(key)\n'*450)}]
    for label,msg in [('text',messages),('vision',[{'role':'user','content':[
        {'type':'text','text':'Read the screenshot. Identify the verification code and correct the Python bug. Explain your reasoning.'},
        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(Path('/home/sebastian/LocalLLM/Local-AI-B70/qwen38/vision/code-review.png').read_bytes()).decode()}}]}])]:
        reset();(data/'enabled').write_text(label)
        result=diag.api('/v1/chat/completions',{'model':'Qwen3.8-27B','messages':msg,'temperature':0,
            'max_tokens':32,'ignore_eos':True,'chat_template_kwargs':{'enable_thinking':False}},timeout=600)
        save(out/(label+'-response.json'),result)
    (data/'enabled').unlink()
    session.stop()
    cmd=['docker','run','--rm','--name',OPERATOR,'--network=none','--device','/dev/dri',
        '--group-add',str(Path('/dev/dri/renderD128').stat().st_gid),'-e','ZE_AFFINITY_MASK=0',
        '-e','ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE','-v',str(ROOT/'e01')+':/src:ro',
        '-v',str(run)+':/work','--entrypoint','python','local/qwen38-b70-vllm:e01-qk-rope-gate-20260919',
        '/src/real-replay.py']
    with (run/'replay.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1500)
