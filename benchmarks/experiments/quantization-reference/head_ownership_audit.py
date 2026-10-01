"""Read-only post-load ownership report; deduplicate target/MTP shared heads."""
import functools
import json
import os
from pathlib import Path

import torch


def inventory(runner):
    assert runner.vllm_config.quant_config.get_name() == 'exl3'
    draft = getattr(getattr(runner, 'speculator', None), 'model', None)
    assert draft is not None
    heads, unique = [], {}
    for role, model in [('target', runner.model), ('draft', draft)]:
        for name, module in model.named_modules():
            report = getattr(module, 'exl3_loader_report', None)
            if report is None or 'lm_head' not in report['members']: continue
            reduced = getattr(module, 'exl3_draft', None)
            buffers = {}
            if reduced is not None:
                for key, tensor in reduced.items():
                    if not isinstance(tensor, torch.Tensor): continue
                    storage = tensor.untyped_storage()
                    identity = (str(tensor.device), storage.data_ptr())
                    record = {'device': identity[0], 'storage_pointer': identity[1],
                              'storage_bytes': storage.nbytes(), 'logical_bytes': tensor.numel() * tensor.element_size()}
                    buffers[key] = record; unique[identity] = record['storage_bytes']
            heads.append({'role': role, 'module': name, 'module_object_id': id(module), 'prefix': report['prefix'],
                          'bits': report['bits'], 'full_rows': module.svh.numel(),
                          'pruned_rows': reduced['svh'].numel() if reduced else None, 'pruned_buffers': buffers})
    assert {h['role'] for h in heads} == {'target', 'draft'} and len(heads) == 2
    return {'heads': heads, 'shared_head_module': heads[0]['module_object_id'] == heads[1]['module_object_id'],
            'unique_pruned_storage_bytes': sum(unique.values()),
            'summed_pruned_role_logical_bytes': sum(v['logical_bytes'] for h in heads for v in h['pruned_buffers'].values()),
            'model_memory_usage': getattr(runner, 'model_memory_usage', None),
            'torch_allocated_bytes': torch.xpu.memory_allocated(), 'torch_reserved_bytes': torch.xpu.memory_reserved(),
            'scope': 'Read-only post-load inventory before KV profiling/graphs. Storage pointers are worker-local identities, not tensor-content hashes. Role totals can double-count a shared target/MTP head.'}


class HeadOwnershipExtension:
    def head_ownership(self):
        return inventory(self.model_runner)


def install():
    if not os.environ.get('EXL3_HEAD_OWNERSHIP_REPORT'): return
    from vllm.v1.worker.gpu_worker import Worker
    assert not getattr(Worker, '_head_ownership_audited', False)
    original = Worker.load_model

    @functools.wraps(original)
    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        path = Path(os.environ['EXL3_HEAD_OWNERSHIP_REPORT'])
        path.write_text(json.dumps(inventory(self.model_runner), indent=2) + '\n')
        return result

    Worker.load_model = load_model
    Worker._head_ownership_audited = True


install()
