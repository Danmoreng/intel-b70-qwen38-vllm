"""Independent FP32 oracle for packed MTP attention around64-token boundaries."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from exl3xpu.shared_kv_verify import run,eligible
sys.path.insert(0,'/opt/exl3xpu/tests')
from test_guarded_attention_xpu import make
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--library',default='/opt/exl3xpu/m04/m04.so');a=p.parse_args()
torch.set_num_threads(1);torch.ops.load_library(a.library)
report=dict(status='RUNNING',runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),torch=torch.__version__,cases=[],rtol=.01,atol=.003,scope='Weights-free M04 packed causal attention vs independently computed CPU FP32; no model/serving qualification.')
def save():a.out.write_text(json.dumps(report,indent=2)+'\n')
def cpu_reference(d):
 q=d['q'].float().cpu();bt=d['block_table'].cpu().long();length=int(d['seqused_k'].cpu()[0]);n=q.shape[0]
 k=d['k'].float().cpu()[bt[0]].reshape(-1,4,256)[:length]*float(d['k_descale'][0,0].cpu())
 v=d['v'].float().cpu()[bt[0]].reshape(-1,4,256)[:length]*float(d['v_descale'][0,0].cpu())
 k=k.repeat_interleave(6,dim=1);v=v.repeat_interleave(6,dim=1)
 scores=torch.einsum('qhd,khd->hqk',q,k)*.0625
 future=torch.arange(length)[None,:]>(length-n+torch.arange(n))[:,None]
 scores.masked_fill_(future[None],-float('inf'))
 return torch.einsum('hqk,khd->qhd',scores.softmax(-1),v)
a.out.parent.mkdir(exist_ok=True,parents=True);save()
for rows in (2,3,4,5):
 for length in (2112,2113,2114,2304,2305,2306,2307,4097):
  torch.manual_seed(2026111400+rows*100+length)
  d,_=make([rows],[length],[False],1792);assert eligible(d)
  expected=cpu_reference(d)
  for splits in ([1,16] if rows==4 else [{2:32,3:8,5:16}[rows]]):
   actual=run(d,splits=splits).float().cpu()
   row=dict(rows=rows,length=length,splits=splits,finite=bool(torch.isfinite(actual).all()),nan_per_query=torch.isnan(actual).sum((1,2)).tolist(),allclose=bool(torch.allclose(actual,expected,rtol=.01,atol=.003)),max_abs=float((actual-expected).abs().max()))
   report['cases'].append(row);save();print('M04_BOUNDARY',row,flush=True)
# Replaying the same graph must handle empty/nonempty causal splits as the
# device length, query values and physical page mapping change.
report['graphs']=[]
torch.manual_seed(2026111417)
d,_=make([4],[2306],[False],1792)
for _ in range(3):run(d)
torch.xpu.synchronize()
graph=torch.xpu.XPUGraph()
with torch.xpu.graph(graph):run(d)
for length in (2306,2307,2304,2114):
 d['seqused_k'].fill_(length);d['q'].mul_(.875)
 d['block_table'].copy_(d['block_table'].flip(1))
 expected=cpu_reference(d)
 graph.replay()
 actual=d['out'].float().cpu()
 row=dict(length=length,finite=bool(torch.isfinite(actual).all()),
          allclose=bool(torch.allclose(actual,expected,rtol=.01,atol=.003)),
          nan_per_query=torch.isnan(actual).sum((1,2)).tolist())
 report['graphs'].append(row);save();print('M04_BOUNDARY_GRAPH',row,flush=True)
report['status']='PASS_M04_BOUNDARY_FP32_ORACLE' if all(c['finite'] and c['allclose'] for c in report['cases']+report['graphs']) else 'FAIL_M04_BOUNDARY_FP32_ORACLE';save()
raise SystemExit(0 if report['status'].startswith('PASS') else 2)
