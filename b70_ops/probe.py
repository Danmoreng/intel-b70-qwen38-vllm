"""Isolate the Q128 temporary-allocation/copy cost on a B70 GPU."""
from pathlib import Path
import json
import random
import statistics
import time
import torch

root=Path("/work")
torch.ops.load_library(str(root/"deployed-tiles.so"))
torch.ops.load_library(str(root/"b70_ops.so"))
torch.manual_seed(42)
base=torch.randn((124,1664,4,2,256),dtype=torch.float16,device="xpu").to(torch.float8_e4m3fn)
k=base[:,:,:,0,:]; v=base[:,:,:,1,:]
assert k.stride()==v.stride()==(3407872,2048,512,1)
block_table=torch.randperm(124,device="xpu",dtype=torch.int64).int()[None,:].contiguous()
k_scale=torch.tensor([0.75],device="xpu"); v_scale=torch.tensor([1.25],device="xpu")

def elapsed(function):
    torch.xpu.synchronize()
    start=torch.xpu.Event(enable_timing=True); end=torch.xpu.Event(enable_timing=True)
    wall=time.perf_counter(); start.record(); returned=function(); end.record(); end.synchronize()
    return {"wall_ms":(time.perf_counter()-wall)*1000,"event_ms":start.elapsed_time(end),
            "returned_ptr":returned.data_ptr()}

checks=[]; rows=[]
for query_tokens,kv_tokens in ((256,8192),(4096,8192),(4096,65536),
                               (4096,131072),(4096,196608),(6656,196608)):
    q=torch.randn((query_tokens,24,256),device="xpu",dtype=torch.float16)
    cu_q=torch.tensor([0,query_tokens],device="xpu",dtype=torch.int32)
    used=torch.tensor([kv_tokens],device="xpu",dtype=torch.int32)
    out=torch.empty_like(q)
    common=(q,k,v,block_table,cu_q,used,k_scale,v_scale)
    reference=torch.ops.b70_tiles.forward(*common,kv_tokens,1)
    returned=torch.ops.b70_ops.q128_forward_out(*common,out,kv_tokens,1)
    torch.xpu.synchronize()
    delta=(out.float()-reference.float()).abs()
    check={"q":query_tokens,"kv":kv_tokens,"finite":bool(out.isfinite().all().cpu()),
           "allclose":bool(torch.allclose(out,reference,rtol=.01,atol=.002)),
           "max_abs":float(delta.max().cpu()),"rms":float(delta.square().mean().sqrt().cpu()),
           "returned_same_storage":returned.data_ptr()==out.data_ptr()}
    checks.append(check); (root/"correctness.json").write_text(json.dumps(checks,indent=2)+"\n")
    print("check",check,flush=True)
    assert check["finite"] and check["allclose"] and check["returned_same_storage"],check
    if query_tokens==256: continue
    def control():
        out.copy_(torch.ops.b70_tiles.forward(*common,kv_tokens,1)); return out
    def direct():
        return torch.ops.b70_ops.q128_forward_out(*common,out,kv_tokens,1)
    functions={"temporary_plus_copy":control,"direct_output":direct}
    for function in functions.values():
        for _ in range(5): function()
    torch.xpu.synchronize(); samples={name:[] for name in functions}
    order=random.Random(query_tokens+kv_tokens)
    for _ in range(31):
        names=list(functions); order.shuffle(names)
        for name in names: samples[name].append(elapsed(functions[name]))
    medians={name:statistics.median(sample["event_ms"] for sample in values)
             for name,values in samples.items()}
    row={"q":query_tokens,"kv":kv_tokens,"output_mib":q.numel()*q.element_size()/2**20,
         "median_event_ms":medians,
         "direct_speedup_pct":100*(medians["temporary_plus_copy"]/medians["direct_output"]-1),
         "samples":samples}
    rows.append(row); (root/"results.json").write_text(json.dumps({"rows":rows},indent=2)+"\n")
    print("timing",{key:value for key,value in row.items() if key!="samples"},flush=True)

(root/"results.json").write_text(json.dumps({
    "scope":"isolated deployed Q128 temporary+copy versus identical Q128 direct output",
    "repeats_per_arm":31,"warmups_per_arm":5,"rows":rows},indent=2)+"\n")
