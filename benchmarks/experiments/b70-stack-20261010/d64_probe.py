"""Bounded untimed same-prefix probe; capture real results before teacher forcing."""
import hashlib,inspect,json
from pathlib import Path
import torch

RUNNERS={'0.30.0':'174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce',
         '0.31.0':'725c47114e148ade3a26b4cc9c5f7295588fdd387afc3cb5fdb2d16513a1bb5c'}
def cpu(x):return x.detach().cpu().clone() if isinstance(x,torch.Tensor) else x
class WorkerExtension:
    def install_d64_probe(self):
        import vllm
        r=self.model_runner
        p=Path(inspect.getfile(type(r)))
        # XPUModelRunnerV2 may live in its thin XPU adapter; verify the
        # actual generic V2 implementation containing sampling/propose calls.
        p=Path(vllm.__file__).parent/'v1/worker/gpu/model_runner.py'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==RUNNERS[vllm.__version__]
        assert r.vllm_config.speculative_config.num_speculative_tokens==3
        assert r.speculator.draft_logits is None,'This probe attests greedy Dirac MTP only'
        assert not r.rejection_sampler.use_block_verification
        assert r.rejection_sampler.synthetic_conditional_rates is None
        self._d64=dict(active=False,folder=None,heads=[],samples=[],processed=[],proposals=[],forced=False,current_capture=False)
        state=self._d64
        original_head=r.model.compute_logits
        def head(*args,**kwargs):
            value=original_head(*args,**kwargs)
            state['current_capture']=False
            if state['active'] and isinstance(value,torch.Tensor) and value.shape[0]>0 and len(state['heads'])<3:
                frame=inspect.currentframe().f_back
                batch=frame.f_locals.get('input_batch')
                assert batch is not None and batch.num_reqs==1
                ix=batch.logits_indices
                positions=cpu(batch.positions[ix])
                if int(positions[-1])<state['prefix_tokens']-1:return value
                state['current_capture']=True
                row=dict(logits=cpu(value),input_ids=cpu(batch.input_ids[ix]),positions=positions,
                    logits_indices=cpu(ix),has_prefill=bool(batch.has_prefill),draft_tokens=int(batch.num_draft_tokens),
                    request_slots=cpu(batch.idx_mapping))
                state['heads'].append(row)
            return value
        r.model.compute_logits=head
        sampler=r.sampler
        original_process=sampler.apply_sampling_params
        def process(*args,**kwargs):
            value=original_process(*args,**kwargs)
            if state['active'] and state['current_capture'] and value.shape[0]>0 and len(state['processed'])<3:
                state['processed'].append(cpu(value))
            return value
        sampler.apply_sampling_params=process
        class SampleCapture:
            def __init__(self,original):self.original=original
            def __getattr__(self,name):return getattr(self.original,name)
            def __call__(self,logits,batch,*args,**kwargs):
                value=self.original(logits,batch,*args,**kwargs)
                if state['active'] and state['current_capture'] and batch.num_reqs==1 and len(state['samples'])<3:
                    ix=batch.logits_indices
                    item=dict(actual_sampled=cpu(value.sampled_token_ids),num_sampled=cpu(value.num_sampled),num_rejected=cpu(value.num_rejected),
                        positions=cpu(batch.positions[ix]),input_ids=cpu(batch.input_ids[ix]),has_prefill=bool(batch.has_prefill),
                        seed=cpu(sampler.sampling_states.seeds.gpu[batch.idx_mapping]),
                        temperature=cpu(sampler.sampling_states.temperature.gpu[batch.idx_mapping]))
                    item['use_fp64_gumbel']=bool(sampler.use_fp64_gumbel)
                    state['samples'].append(item)
                    # One anchor is teacher-forced only AFTER the real target
                    # logits, sampling transforms and native RNG result were
                    # captured. Both versions then build their own MTP state
                    # from the identical prefix + anchor. Later outputs remain
                    # native; no frozen logits/state replace computed results.
                    if not state['forced']:
                        assert int(value.num_sampled[0])==1
                        assert int(batch.positions[ix][-1])==state['prefix_tokens']-1
                        value.sampled_token_ids[0,0]=state['anchor']
                        item['teacher_forced_anchor']=state['anchor'];state['forced']=True
                return value
        r.sampler=SampleCapture(r.sampler)
        if r.rejection_sampler is not None:r.rejection_sampler=SampleCapture(r.rejection_sampler)
        original_propose=r.speculator.propose;sig=inspect.signature(original_propose)
        def propose(*args,**kwargs):
            value=original_propose(*args,**kwargs)
            if state['active'] and state['forced'] and len(state['proposals'])<2:
                bound=sig.bind(*args,**kwargs).arguments
                batch=bound.get('input_batch')
                assert batch is not None and batch.num_reqs==1
                state['proposals'].append(dict(tokens=cpu(value),sample_src_positions=cpu(r.speculator.sample_src_positions[:1]),
                    method='greedy',q_semantics='Dirac at each actual autoregressive proposal',
                    conditioning='First step shares the forced anchor; later steps condition on this version\'s earlier actual proposals'))
            return value
        r.speculator.propose=propose
        return dict(status='INSTALLED_BOUNDED_TEACHER_FORCED_DIAGNOSTIC',vllm=vllm.__version__,runner_sha256=RUNNERS[vllm.__version__])
    def arm_d64_probe(self,label,prefix_tokens,anchor):
        state=self._d64;assert not state['active'];folder=Path('/results/d64')/label;folder.mkdir(parents=True,exist_ok=False)
        state.update(active=True,folder=str(folder),heads=[],samples=[],processed=[],proposals=[],forced=False,current_capture=False,prefix_tokens=prefix_tokens,anchor=anchor)
        return {'status':'ARMED','prefix_tokens':prefix_tokens,'anchor':anchor}
    def finish_d64_probe(self):
        state=self._d64;state['active']=False
        assert state['forced'] and len(state['heads'])>=2 and state['proposals']
        assert len(state['processed'])==len(state['samples'])==len(state['heads'])
        payload={k:v for k,v in state.items() if k not in ('active','folder')}
        torch.save(payload,Path(state['folder'])/'capture.pt')
        report=dict(status='COMPLETE_SAME_PREFIX_ACTUAL_RESULT_CAPTURE_NOT_BENCHMARK',
            prefix_tokens=state['prefix_tokens'],anchor=state['anchor'],head_rows=[list(x['logits'].shape) for x in state['heads']],
            head_logits_sha256=[hashlib.sha256(x['logits'].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for x in state['heads']],
            actual_samples=[x['actual_sampled'].tolist() for x in state['samples']],
            proposals=[x['tokens'].tolist() for x in state['proposals']],
            scope='Own-version states reconstructed from frozen prompt IDs; one shared anchor teacher-forced after native-result capture. No process-specific state/graphs transplanted or repeated shadow replay. Later draft-branch divergence is reported as differing conditioning, not same-prefix numerical error.')
        (Path(state['folder'])/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
