"""Correctness and alternating timing for the first GPTQ small-M challenger."""
from pathlib import Path
import json,random,statistics,time,torch
from vllm_xpu_kernels import _xpu_C  # noqa: F401 -- registers torch.ops._xpu_C

root=Path("/work"); torch.ops.load_library(str(root/"b70_ops.so")); torch.manual_seed(17)
rows=[]; checks=[]

def timed(function):
    torch.xpu.synchronize(); start=torch.xpu.Event(enable_timing=True); end=torch.xpu.Event(enable_timing=True)
    wall=time.perf_counter(); start.record(); value=function(); end.record(); end.synchronize()
    return {"event_ms":start.elapsed_time(end),"wall_ms":(time.perf_counter()-wall)*1000}

for m,k,n in ((1,5120,34816),(3,5120,34816),(5,5120,34816),
              (1,5120,14336),(5,5120,14336),(1,17408,5120),(5,17408,5120)):
    x=torch.randn((m,k),device="xpu",dtype=torch.float16)
    storage=torch.randint(-(2**31),2**31-1,(n,k//8),device="xpu",dtype=torch.int32)
    weight=storage.t(); assert weight.stride()==(1,k//8)
    scales=(torch.rand((k//128,n),device="xpu",dtype=torch.float16)*0.02+0.0001).contiguous()
    zero=torch.tensor([8],device="xpu",dtype=torch.int8)
    def control(): return torch.ops._xpu_C.int4_gemm_w4a16(x,weight,None,scales,zero,128,None)
    def challenger(): return torch.ops.b70_ops.gptq_small_m(x,weight,scales,zero,128)
    expected=control(); actual=challenger(); torch.xpu.synchronize()
    delta=(expected.float()-actual.float()).abs()
    check={"m":m,"k":k,"n":n,"finite":bool(actual.isfinite().all().cpu()),
           "allclose":bool(torch.allclose(actual,expected,rtol=.01,atol=.02)),
           "max_abs":float(delta.max().cpu()),"rms":float(delta.square().mean().sqrt().cpu())}
    checks.append(check); (root/"correctness.json").write_text(json.dumps(checks,indent=2)+"\n")
    print("check",check,flush=True)
    if not check["finite"] or not check["allclose"]: continue
    functions={"onednn_w4a16":control,"b70_scalar_reuse":challenger}
    for function in functions.values():
        for _ in range(5): function()
    samples={name:[] for name in functions}; rng=random.Random(m+k+n)
    for _ in range(31):
        names=list(functions); rng.shuffle(names)
        for name in names: samples[name].append(timed(functions[name]))
    medians={name:statistics.median(v["event_ms"] for v in values) for name,values in samples.items()}
    row={"m":m,"k":k,"n":n,"median_event_ms":medians,
         "challenger_speedup_pct":100*(medians["onednn_w4a16"]/medians["b70_scalar_reuse"]-1),
         "samples":samples}
    rows.append(row); (root/"results.json").write_text(json.dumps({"rows":rows},indent=2)+"\n")
    print("timing",{key:value for key,value in row.items() if key!="samples"},flush=True)

(root/"results.json").write_text(json.dumps({"scope":"same packed GPTQ W4A16 coefficients; oneDNN control versus row-reuse scalar SYCL challenger","rows":rows},indent=2)+"\n")
