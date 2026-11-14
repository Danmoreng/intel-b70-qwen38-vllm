"""Opt-in diagnostic trace for the pinned EXL3 runtimes. No production dispatch changes.

These runs are for memory/accounting diagnosis, not unbiased throughput timing.
Unknown scheduler/cache source fails closed; vLLM 0.31 callbacks preserve native behavior.
"""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import torch
import vllm

OLD_SOURCE_HASHES={
 'v1/core/sched/scheduler.py':'2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941',
 'v1/core/kv_cache_manager.py':'3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf',
 'v1/core/single_type_kv_cache_manager.py':'bcb27e38895332bf6a4c55608f2917eb9fd941ec5a629c14b88358f00aadeba8',
 'v1/worker/gpu_worker.py':'28ac3f85aca3f68f18ac51cb5915e4d7ebe316c4ff3c021f1c2c5d3b90daca56',
}
TARGET_SOURCE_HASHES={
 'v1/core/sched/scheduler.py':'c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884',
 'v1/core/kv_cache_manager.py':'58800535eda854580d84e1025bf32fe01b2a43ce0bba501b4f3d125e4f2c8afa',
 'v1/core/single_type_kv_cache_manager.py':'0233531d85aeced9e57a38c76d83db970519c92902b7e55484390cb2b64985bf',
 'v1/worker/gpu_worker.py':'6994436e4547c0996ab555b3e814e8ccea26b650dfc1b378b97d640745309347',
}

V031_SOURCE_HASHES={'v1/core/sched/scheduler.py': 'f820bee4f42b878b262a30f3728a97d7413477b2a277748e20f2528ac2ec2f2a',
 'v1/core/kv_cache_manager.py': 'bd9bf46a2101f9368d4941867415a239d18d95e0b956f82f885ce25e10e69f3b',
 'v1/core/single_type_kv_cache_manager.py': '3a3cb20bd6aa1e0ad46471fc9e5e01ee5797ab4bb759e00ef5fb309d2830db13',
 'v1/worker/gpu_worker.py': '43bd01925cd84f7d8a1d4a2ec30e289014a485fe4492a5343cbc893e8ed2f518'}

def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def memory():
    if not torch.xpu.is_initialized(): return {'available':False,'reason':'no XPU context in this process'}
    result={'allocated':torch.xpu.memory_allocated(),'reserved':torch.xpu.memory_reserved(),
            'max_allocated':torch.xpu.max_memory_allocated(),'max_reserved':torch.xpu.max_memory_reserved(),
            'oneDNN_workspace_bytes':None,'graph_pool_bytes':None}
    if hasattr(torch.xpu,'mem_get_info'):
        try: result['free'],result['total']=torch.xpu.mem_get_info()
        except RuntimeError as e: result['free_query_error']=str(e)
    return result

def tensor_storage(model):
    """Host metadata only. Count shared parameter/buffer/draft-head storage once."""
    if model is None: return None
    models=model if isinstance(model,list) else [model]
    seen=set();by_dtype={};named_bytes=0;extra_bytes=0
    def record(tensor,extra=False):
        nonlocal named_bytes,extra_bytes
        if not isinstance(tensor,torch.Tensor) or tensor.device.type!='xpu': return
        storage=tensor.untyped_storage();key=(str(tensor.device),storage.data_ptr())
        if key in seen: return
        seen.add(key);n=storage.nbytes();by_dtype[str(tensor.dtype)]=by_dtype.get(str(tensor.dtype),0)+n
        if extra: extra_bytes+=n
        else: named_bytes+=n
    for model in models:
        for tensor in list(model.parameters())+list(model.buffers()): record(tensor)
        for layer in model.modules():
            for tensor in getattr(layer,'exl3_draft',{}).values(): record(tensor,True)
    return {'unique_storage_bytes':named_bytes+extra_bytes,'named_parameter_buffer_bytes':named_bytes,
            'extra_pruned_draft_head_bytes':extra_bytes,'storage_count':len(seen),'bytes_by_dtype':by_dtype,
            'scope':'named XPU parameters/buffers and explicit exl3_draft tensor dictionaries; workspaces are excluded'}

def emit(event,**fields):
    root=Path(os.environ['EXL3_MIGRATION_TRACE']);root.mkdir(parents=True,exist_ok=True)
    record={'event':event,'pid':os.getpid(),'unix':time.time(),'monotonic':time.monotonic(),**fields}
    with (root/f'trace-{os.getpid()}.jsonl').open('a') as f:
        f.write(json.dumps(record,default=str,separators=(',',':'))+'\n')

def request_summary(request):
    result={n:getattr(request,n,None) for n in ('request_id','num_prompt_tokens','num_tokens',
             'num_computed_tokens','num_cached_tokens','num_preemptions','num_in_flight_tokens','status')}
    result['status_name']=getattr(getattr(request,'status',None),'name',None)
    return result

def cache_snapshot(manager,request_id=None):
    groups=[]
    for i,m in enumerate(manager.coordinator.single_type_managers):
        blocks=m.req_to_blocks.get(request_id,[]) if request_id is not None else []
        groups.append({'group':i,'type':type(m).__name__,'spec':repr(m.kv_cache_spec),
                       'block_size':m.block_size,'num_speculative_blocks':getattr(m,'num_speculative_blocks',None),
                       'held_blocks':len(blocks),'non_null_blocks':sum(b.block_id!=0 for b in blocks),
                       'block_ids':[b.block_id for b in blocks],
                       'ref_counts':[b.ref_cnt for b in blocks]})
    return {'total_blocks':manager.block_pool.num_gpu_blocks,
            'free_blocks':manager.block_pool.get_num_free_blocks(),
            'watermark_blocks':manager.watermark_blocks,'groups':groups}

def native_map():
    paths=set()
    for row in Path('/proc/self/maps').read_text().splitlines():
        parts=row.split()
        if len(parts)>=6 and parts[-1].startswith('/') and '.so' in parts[-1] and Path(parts[-1]).is_file(): paths.add(parts[-1])
    return {p:{'bytes':Path(p).stat().st_size,'sha256':digest(p)} for p in sorted(paths)}

def wrap(cls,name,callback):
    original=getattr(cls,name)
    @functools.wraps(original)
    def wrapped(self,*args,**kw): return callback(original,self,args,kw)
    setattr(cls,name,wrapped)

def install():
    if not os.environ.get('EXL3_MIGRATION_TRACE'): return
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu_worker import Worker
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    if getattr(KVCacheManager,'_exl3_migration_traced',False): return
    root=Path(vllm.__file__).parent
    runtime=os.environ.get('EXL3_TARGET_RUNTIME')
    expected=(V031_SOURCE_HASHES if runtime=='vllm031' else
              TARGET_SOURCE_HASHES if runtime=='vllm030' else OLD_SOURCE_HASHES)
    actual={n:digest(root/n) for n in expected}
    if actual!=expected: raise RuntimeError(f'Unknown runtime for EXL3 diagnostic trace: {actual}')
    emit('trace_install',vllm_version=vllm.__version__,torch_version=torch.__version__,
         source_hashes=actual,sparse_gdn_retirement_fix=bool(getattr(MambaManager,'_exl3_sparse_retirement_fixed',False)),
         gdn_retirement_method_source=inspect.getsourcefile(MambaManager._remove_blocks_in_range),
         gdn_retirement_method_sha256=hashlib.sha256(inspect.getsource(MambaManager._remove_blocks_in_range).encode()).hexdigest(),
         note='instrumented diagnostics; throughput is not a release benchmark')

    def init_cache(original,self,args,kw):
        result=original(self,*args,**kw)
        emit('cache_initialized',cache=cache_snapshot(self))
        for i,m in enumerate(self.coordinator.single_type_managers):
            bound=m.get_num_blocks_to_allocate
            signature=inspect.signature(bound)
            def traced_query(*args,_bound=bound,_signature=signature,_manager=m,_group=i,**kw):
                inputs=_signature.bind(*args,**kw);inputs.apply_defaults()
                data=inputs.arguments
                requested=_bound(*args,**kw)
                emit('group_allocation_query',group=_group,type=type(_manager).__name__,
                     request_id=data.get('request_id'),num_tokens=data.get('num_tokens'),
                     num_tokens_main_model=data.get('num_tokens_main_model'),
                     total_computed_tokens=data.get('total_computed_tokens'),
                     admission_cap=data.get('apply_admission_cap'),requested_blocks=requested,
                     free_pool_blocks=self.block_pool.get_num_free_blocks(),
                     prefix_computed_blocks=len(data.get('new_computed_blocks',[])))
                return requested
            m.get_num_blocks_to_allocate=traced_query
        return result
    wrap(KVCacheManager,'__init__',init_cache)
    allocate_signature=inspect.signature(KVCacheManager.allocate_slots)
    def allocate(original,self,args,kw):
        bound=allocate_signature.bind(self,*args,**kw);bound.apply_defaults();data=bound.arguments
        request=data['request'];rid=request.request_id
        emit('before_allocation',request=request_summary(request),cache=cache_snapshot(self,rid),
             arguments={k:v for k,v in data.items() if k not in ('self','request','new_computed_blocks')})
        result=original(self,*args,**kw)
        emit('allocation_rejected' if result is None else 'allocation_completed',
             request=request_summary(request),cache=cache_snapshot(self,rid),
             memory=memory() if result is None else None)
        return result
    wrap(KVCacheManager,'allocate_slots',allocate)

    def admit(original,self,args,kw):
        request=args[0] if args else kw['request']
        emit('request_admission',request=request_summary(request),memory=memory(),
             cache=cache_snapshot(self.kv_cache_manager),lookahead_tokens=self.num_lookahead_tokens)
        return original(self,*args,**kw)
    wrap(Scheduler,'add_request',admit)
    def preempt(original,self,args,kw):
        request=args[0] if args else kw['request']
        emit('before_preemption',request=request_summary(request),cache=cache_snapshot(self.kv_cache_manager,request.request_id),
             memory=memory(),reason='Scheduler._preempt_request; correlate immediately preceding allocation_rejected (or explicit preempt API)')
        result=original(self,*args,**kw)
        emit('after_preemption_requeue',request=request_summary(request),cache=cache_snapshot(self.kv_cache_manager,request.request_id),memory=memory())
        return result
    wrap(Scheduler,'_preempt_request',preempt)
    def free_request(original,self,args,kw):
        request=args[0] if args else kw['request']
        emit('request_finish_or_abort',request=request_summary(request),cache=cache_snapshot(self.kv_cache_manager,request.request_id),memory=memory())
        return original(self,*args,**kw)
    wrap(Scheduler,'_free_request',free_request)
    def schedule(original,self,args,kw):
        before={rid:request_summary(r) for rid,r in self.requests.items()}
        result=original(self,*args,**kw)
        if result.num_scheduled_tokens:
            emit('scheduled_work',tokens=result.num_scheduled_tokens,request_state_before=before,
                 note='scheduled work including speculation, not logical prompt metric or proof of completed GPU compute')
        return result
    wrap(Scheduler,'schedule',schedule)

    def worker_load(original,self,args,kw):
        emit('before_weights',memory=memory())
        runner=self.model_runner
        proposer=getattr(runner,'speculator',None) or getattr(runner,'drafter',None)
        if proposer is not None and hasattr(proposer,'load_model') and not getattr(type(proposer),'_exl3_draft_load_traced',False):
            def draft_load(original,self,args,kw):
                target=args[0] if args else kw.get('target_model')
                emit('before_draft_weights',memory=memory(),target_tensor_storage=tensor_storage(target))
                result=original(self,*args,**kw)
                draft=getattr(self,'model',None)
                language=target.get_language_model() if hasattr(target,'get_language_model') else target
                head=getattr(language,'lm_head',None)
                emit('after_draft_weights',memory=memory(),draft_tensor_storage=tensor_storage(draft),
                     proposer_class=type(self).__name__,
                     shared_lm_head=None if head is None else head is getattr(draft,'lm_head',None))
                return result
            wrap(type(proposer),'load_model',draft_load)
            type(proposer)._exl3_draft_load_traced=True
        result=original(self,*args,**kw)
        runner=self.model_runner
        proposer=getattr(runner,'speculator',None) or getattr(runner,'drafter',None)
        models=[runner.model]+([proposer.model] if proposer is not None and hasattr(proposer,'model') else [])
        from .memory_audit import unowned_tensor_audit
        emit('after_weights_and_draft',memory=memory(),runner_class=type(runner).__name__,
             runner_source=inspect.getfile(type(runner)),
             runner_mro=[str(c) for c in type(runner).__mro__],model_tensor_storage=tensor_storage(models),
             unowned_tensor_audit=unowned_tensor_audit(models),
             draft_present=(getattr(runner,'speculator',None) or getattr(runner,'drafter',None)) is not None)
        runner_class=type(runner)
        if not getattr(runner_class,'_exl3_capture_traced',False):
            def capture(original,self,args,kw):
                emit('before_graph_capture',memory=memory())
                result=original(self,*args,**kw)
                emit('after_graph_capture',memory=memory(),reported_capture_bytes=result)
                return result
            wrap(runner_class,'capture_model',capture)
            runner_class._exl3_capture_traced=True
        return result
    wrap(Worker,'load_model',worker_load)
    def profile_memory(original,self,args,kw):
        emit('before_memory_profile',memory=memory(),requested_memory=getattr(self,'requested_memory',None))
        result=original(self,*args,**kw)
        emit('after_memory_profile',memory=memory(),available_kv_cache_memory_bytes=result,
             requested_memory=getattr(self,'requested_memory',None),
             total_consumed=getattr(self,'total_consumed',None),
             peak_activation_memory=getattr(self,'peak_activation_memory',None),
             cudagraph_memory_estimate=getattr(self,'cudagraph_memory_estimate',None),
             model_memory_usage=getattr(self.model_runner,'model_memory_usage',None))
        return result
    wrap(Worker,'determine_available_memory',profile_memory)
    def warm(original,self,args,kw):
        result=original(self,*args,**kw)
        emit('after_warmup',memory=memory(),native_libraries=native_map())
        return result
    wrap(Worker,'compile_or_warm_up_model',warm)
    def execute(original,self,args,kw):
        output=args[0] if args else kw.get('scheduler_output')
        if output is None: return original(self,*args,**kw)
        new_requests=getattr(output,'scheduled_new_reqs',[])
        cached=getattr(output,'scheduled_cached_reqs',None)
        offsets={getattr(r,'req_id',None):getattr(r,'num_computed_tokens',None) for r in new_requests}
        if cached is not None:
            offsets.update(zip(getattr(cached,'req_ids',[]),getattr(cached,'num_computed_tokens',[])))
        emit('target_forward_submission',tokens=output.num_scheduled_tokens,computed_offsets=offsets,
             runner_class=type(self.model_runner).__name__)
        result=original(self,*args,**kw)
        emit('target_forward_return',tokens=output.num_scheduled_tokens,
             note='worker returned after submitting forward work; GPU completion is asynchronous')
        return result
    wrap(Worker,'execute_model',execute)
    KVCacheManager._exl3_migration_traced=True
