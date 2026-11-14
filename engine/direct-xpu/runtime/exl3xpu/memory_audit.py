"""Startup-only host inspection of live XPU tensors outside the loaded models."""
import gc
import torch


def unowned_tensor_audit(models):
    known=set();modules=set()
    for model in models:
        for layer in model.modules():
            modules.add(id(layer))
            for t in list(layer.parameters(recurse=False))+list(layer.buffers(recurse=False)):
                if t.device.type=='xpu': known.add(t.untyped_storage().data_ptr())
            for value in vars(layer).values():
                tensors=[value] if isinstance(value,torch.Tensor) else value.values() if isinstance(value,dict) else []
                for t in tensors:
                    if isinstance(t,torch.Tensor) and t.device.type=='xpu': known.add(t.untyped_storage().data_ptr())
    objects=gc.get_objects();seen=set();rows=[];unowned_heads=[]
    for obj in objects:
        if isinstance(obj,torch.Tensor) and obj.device.type=='xpu':
            storage=obj.untyped_storage();ptr=storage.data_ptr()
            if ptr in known or ptr in seen or storage.nbytes()<1024*1024: continue
            seen.add(ptr)
            refs=[]
            for ref in gc.get_referrers(obj):
                if ref is objects: continue
                summary={'type':type(ref).__name__}
                if isinstance(ref,dict):
                    summary['tensor_keys']=[str(k) for k,v in list(ref.items()) if v is obj][:12]
                refs.append(summary)
            rows.append({'shape':list(obj.shape),'dtype':str(obj.dtype),'bytes':storage.nbytes(),
                         'tensor_type':type(obj).__name__,'referrers':refs[:10]})
        elif isinstance(obj,torch.nn.Module) and id(obj) not in modules:
            report=getattr(obj,'exl3_loader_report',None)
            if report and 'lm_head' in report['members']:
                refs=[]
                for ref in gc.get_referrers(obj):
                    if ref is objects: continue
                    summary={'type':type(ref).__name__}
                    if isinstance(ref,dict): summary['module_keys']=[str(k) for k,v in list(ref.items()) if v is obj][:12]
                    if type(ref).__name__=='method': summary['method']=getattr(ref,'__name__',None)
                    refs.append(summary)
                unowned_heads.append({'class':type(obj).__name__,'prefix':report['prefix'],'referrers':refs[:12]})
    rows.sort(key=lambda r:r['bytes'],reverse=True)
    return {'unowned_storage_bytes_over_1MiB':sum(r['bytes'] for r in rows),'largest_tensors':rows[:24],
            'unowned_exl3_heads':unowned_heads,'scope':'Python-GC-visible XPU storage not referenced by model parameters/buffers/direct attributes/dictionaries; host metadata only'}
