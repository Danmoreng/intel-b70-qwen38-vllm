"""Fail-closed whole-model inventory check after target and MTP weight loading."""
import functools
import json
import os
from pathlib import Path
from collections import Counter
from vllm.logger import init_logger
logger=init_logger('vllm.exl3xpu.loader')

def audit_runner(runner):
    config=getattr(runner.vllm_config,'quant_config',None)
    if config is None or config.get_name()!='exl3': return None
    expected={v['canonical']:v for v in config.expected_inventory}
    proposer=getattr(runner,'speculator',None) or getattr(runner,'drafter',None)
    draft=getattr(proposer,'model',None)
    models=[('target',runner.model)]+([('draft',draft)] if draft is not None else [])
    reports=[];counts=Counter()
    for role,model in models:
        seen=set()
        for name,layer in model.named_modules():
            if id(layer) in seen: continue
            seen.add(id(layer))
            record=getattr(layer,'exl3_loader_report',None)
            if record is None: continue
            reports.append(dict(record,role=role,module=name))
            for member in record['members']:
                if member not in expected or record['bits']!=expected[member]['bits']:
                    raise ValueError(f'EXL3 unexpected loaded tensor/bit width: {member}')
                counts[member]+=1
    required={k for k in expected if not k.startswith('mtp.') or draft is not None}
    missing=required-set(counts)
    duplicates={k:v for k,v in counts.items() if v>1 and k!='lm_head'}
    result={'schema':1,'pid':os.getpid(),'status':'FAIL' if missing or duplicates else 'PASS',
            'expected_modules':len(required),'loaded_modules':len(counts),'missing':sorted(missing),
            'unexpected_duplicates':duplicates,'shared_head_load_count':counts.get('lm_head',0),
            'mtp_expected':sum(k.startswith('mtp.') for k in required),
            'mtp_loaded':sum(k.startswith('mtp.') for k in counts),'layers':reports,
            'mtp_skipped_reason':None if draft is not None else 'speculation disabled; no draft model instantiated'}
    path=os.environ.get('EXL3_LOADER_REPORT_DIR')
    if path:
        Path(path).mkdir(parents=True,exist_ok=True)
        (Path(path)/f'loader-{os.getpid()}.json').write_text(json.dumps(result,indent=2)+'\n')
    if missing or duplicates: raise ValueError(f'EXL3 loader mismatch: missing={sorted(missing)}, duplicates={duplicates}')
    logger.info('EXL3 loader audit PASS: %d/%d modules; %d MTP; exact shapes/groups verified',
                len(counts),len(required),result['mtp_loaded'])
    return result

def install():
    # XPUWorker inherits this method for both V1 and V2 runners. Auditing only
    # GPUModelRunner(V1) would silently miss the active V2 serving path.
    from vllm.v1.worker.gpu_worker import Worker
    if getattr(Worker,'_exl3_loader_audited',False): return
    original=Worker.load_model
    @functools.wraps(original)
    def load_model(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        audit_runner(self.model_runner)
        return result
    Worker.load_model=load_model
    Worker._exl3_loader_audited=True
