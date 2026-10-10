"""Read-only worker inventory; called outside timed serving waves."""
import ctypes
import hashlib
import inspect
import json
from pathlib import Path
import torch
from profile_capture import ProfileWorkerExtension

def plain(value):
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return str(value)

class WorkerExtension(ProfileWorkerExtension):
    def inventory(self,label='initial',expected_onednn_sha256='ee8fc42adcf851ea0f8013ebda24f07629f49ca6ed815835a16da45ad718ca20',expected_onednn_version='3.13.0'):
        r=self.model_runner;c=r.vllm_config
        assert type(r).__name__=='XPUModelRunnerV2'
        assert c.speculative_config.method=='mtp' and c.speculative_config.num_speculative_tokens==3
        assert c.model_config.dtype==torch.float16 and c.model_config.max_model_len==262144
        mapped=[]
        for line in Path('/proc/self/maps').read_text().splitlines():
            if 'libdnnl' in line:
                p=Path(line.split()[-1])
                if str(p) not in [i['path'] for i in mapped]:
                    mapped.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        assert len(expected_onednn_sha256)==64
        assert any(i['path']=='/opt/b70/libdnnl.so.3' and i['sha256']==expected_onednn_sha256 for i in mapped),mapped
        class Version(ctypes.Structure):
            _fields_=[('major',ctypes.c_int),('minor',ctypes.c_int),('patch',ctypes.c_int),('hash',ctypes.c_char_p)]
        lib=ctypes.CDLL('/opt/b70/libdnnl.so.3');lib.dnnl_version.restype=ctypes.POINTER(Version);v=lib.dnnl_version().contents
        assert f'{v.major}.{v.minor}.{v.patch}'==expected_onednn_version
        sources={};modules=[];tensor_dtypes={}
        for role,model in [('target',r.model),('mtp',r.speculator.model)]:
            counts={}
            for kind,tensors in [('parameter',model.parameters()),('buffer',model.buffers())]:
                for tensor in tensors:
                    key=kind+':'+str(tensor.dtype)
                    counts[key]=counts.get(key,0)+1
            tensor_dtypes[role]=counts
            for name,m in model.named_modules():
                try:p=Path(inspect.getfile(type(m)))
                except (TypeError,OSError):continue
                if p.is_file() and p.suffix=='.py':sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                if any(s in type(m).__name__.lower() for s in ['gated','gdn','attention','exl3']):
                    modules.append(dict(role=role,name=name,type=type(m).__name__,source=str(p),
                        attributes={k:plain(getattr(m,k)) for k in ['num_heads','num_kv_heads','num_v_heads','head_dim','eps','norm_before_gate','activation','group_size'] if hasattr(m,k)}))
        unique={}
        for t in r.kv_caches:
            st=t.untyped_storage();unique[(str(t.device),st.data_ptr())]=st.nbytes()
        cache=r.kv_cache_config;pool=sum(unique.values())
        groups=[dict(index=i,layers=g.layer_names,spec_type=type(g.kv_cache_spec).__name__,
            page_size_bytes=g.kv_cache_spec.page_size_bytes,
            max_request_bytes=g.kv_cache_spec.max_memory_usage_bytes(c),host_resident=g.host_resident)
            for i,g in enumerate(cache.kv_cache_groups)]
        free,total=torch.xpu.mem_get_info()
        stats=torch.ops.exl3xpu_C.exl3_sdpa_cache_stats(torch.empty(0,device='xpu',dtype=torch.float16))
        result=dict(status='ACTUAL_WORKER_READ_ONLY_INVENTORY',runner=type(r).__name__,speculator=type(r.speculator).__name__,
            custom_oneDNN=dict(version=f'{v.major}.{v.minor}.{v.patch}',git=v.hash.decode(),mapped_libraries=mapped),
            configured_max_model_len=c.model_config.max_model_len,mtp_tokens=c.speculative_config.num_speculative_tokens,
            kv_pool_bytes=pool,num_blocks=cache.num_blocks,pool_stride_bytes=pool//cache.num_blocks,groups=groups,
            torch_allocated_bytes=torch.xpu.memory_allocated(),torch_reserved_bytes=torch.xpu.memory_reserved(),
            peak_allocated_bytes=torch.xpu.max_memory_allocated(),peak_reserved_bytes=torch.xpu.max_memory_reserved(),
            free_bytes=free,total_bytes=total,sdpa_partition_cache_stats=dict(stats),sources=sources,modules=modules,
            registered_tensor_dtypes=tensor_dtypes,
            profiler_activities=[str(a) for a in torch.profiler.supported_activities()],
            route_scope='Model objects and sources establish eligibility, not captured execution. Kernel trace supplies graph-route evidence.',
            graph_pool_scope='Graph and native workspaces are included in allocator/device residency; independently attributed only where a later diagnostic proves ownership.')
        Path(f'/results/inventory-{label}.json').write_text(json.dumps(result,indent=2)+'\n')
        return {k:v for k,v in result.items() if k not in ('sources','modules','groups')}

    def begin_bounded_trace(self,cycles=8,min_requests=1):
        result=self.begin_kernel_trace(cycles)
        self.model_runner._exl3_event_capture.trace_min_requests=min_requests
        return {**result,'min_requests':min_requests}
