"""Perturbing eager-only branch/KV capture. Never attach to timing workers."""
import inspect
from pathlib import Path
import torch
from diagnostics import WorkerExtension as Identity

def cpu(x):
    return x.detach().float().cpu() if isinstance(x,torch.Tensor) else x

class WorkerExtension(Identity):
    def epsilon_probe(self):
        model=self.model_runner.speculator.model.model
        norms=[(f'layer{i}.{name}',getattr(layer,name)) for i,layer in enumerate(model.layers)
            for name in ['input_layernorm','post_attention_layernorm']]+[('final',model.norm)]
        rows=[]
        pattern=((torch.arange(8*5120,device='xpu').reshape(8,5120)%17)-8).float()/8
        for name,norm in norms:
            assert 0<norm.variance_epsilon<1e-8
            for amplitude in [1e-5,1e-3]:
                x=(pattern*amplitude).half();actual=cpu(norm(x))
                value=cpu(x).double()
                expected=value*torch.rsqrt(value.square().mean(-1,keepdim=True)+norm.variance_epsilon)*cpu(norm.weight).double()
                assert torch.allclose(actual.double(),expected,rtol=.01,atol=.005),(name,amplitude)
                rows.append(dict(norm=name,amplitude=amplitude,epsilon=norm.variance_epsilon,
                    max_abs=float((actual.double()-expected).abs().max()),rtol=.01,atol=.005))
        import json
        result=dict(status='PASS_ACTUAL_FP16_SMALL_EPSILON_NORMS',checks=rows,
            scope='Actual eager norms with small inputs where scaled epsilon is significant; graph-fused arithmetic remains separately scoped')
        Path('/results/epsilon-probe.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    def install_branch_capture(self):
        r=self.model_runner;s=r.speculator;model=s.model.model
        assert r.vllm_config.model_config.enforce_eager
        assert not s.enable_adaptive_verification
        state=dict(round=0,selected={2,16,48,80},active=False,current=None,files=[])
        folder=Path('/results/branch-capture');folder.mkdir()
        target=r.model.get_language_model() if hasattr(r.model,'get_language_model') else r.model
        core=target.model;features={}
        original_aux=core._maybe_add_hidden_state
        auxsig=inspect.signature(original_aux)
        def aux(*args,**kwargs):
            bound=auxsig.bind(*args,**kwargs).arguments
            value=original_aux(*args,**kwargs)
            layer=bound.get('layer_idx',bound.get('layer_id'))
            if layer in core.aux_hidden_state_layers:
                hs=bound['hidden_states'];res=bound.get('residual')
                outputs=bound.get('aux_hidden_states')
                assert outputs
                # Subsequent fused norms mutate the residual buffer. Preserve
                # the actual post-layer operands in this untimed eager pass.
                features[layer]=(hs[-8:].clone(),res[-8:].clone() if res is not None else None,outputs[-1][-8:].clone())
            return value
        core._maybe_add_hidden_state=aux
        original_combine=s.model.combine_hidden_states
        def combine(value):
            result=original_combine(value)
            if state['active']:
                columns=list(range(0,5120,256))+[5119]
                state['current']['feature_fc']=dict(input=cpu(value),columns=columns,
                    weight=cpu(model.fc.weight[columns]),output=cpu(result[:,columns]))
            return result
        s.model.combine_hidden_states=combine
        original_project=model._project_context_kv
        def project(context,num_ctx,num_layers,num_kv_heads,head_dim):
            result=original_project(context,num_ctx,num_layers,num_kv_heads,head_dim)
            if state['active']:
                assert model._fused_kv_weight is not None
                # Selected columns across K/V and all five layers. This
                # references original FP16 coefficients, not another GEMM.
                columns=[i*(2*model._kv_size)+j for i in range(5)
                    for j in [0,127,model._kv_size-1,model._kv_size,model._kv_size+127,2*model._kv_size-1]]
                state['current']['context_projection']=dict(input=cpu(context),
                    norm_weight=cpu(model._hidden_norm_weight),eps=model._rms_norm_eps,
                    columns=columns,weight=cpu(model._fused_kv_weight[columns]),
                    k=cpu(result[0]),v=cpu(result[1]))
            return result
        model._project_context_kv=project
        for index,layer in enumerate(model.layers):
            attn=layer.self_attn.attn;original_update=attn.impl.do_kv_cache_update
            def update(a,key,value,cache,slots,fn=original_update,layer_index=index):
                result=fn(a,key,value,cache,slots)
                if state['active'] and layer_index==0:
                    hd=128;kc,vc=cache.transpose(1,2).split(hd,-1)
                    kc=kc.view(torch.float8_e4m3fn);vc=vc.view(torch.float8_e4m3fn)
                    valid=slots>=0;selected=slots[valid].long();page=kc.shape[1]
                    state['current']['cache_writes'].append(dict(layer=layer_index,key=cpu(key),value=cpu(value),
                        slots=slots.detach().cpu(),valid=valid.detach().cpu(),
                        k_scale=cpu(a._k_scale),v_scale=cpu(a._v_scale),
                        stored_k=cpu(kc[selected//page,selected%page]),
                        stored_v=cpu(vc[selected//page,selected%page])))
                return result
            attn.impl.do_kv_cache_update=update
        # These wrappers run only in eager diagnostics; capture a few rows
        # rather than synchronizing tensors in a benchmark graph replay.
        for name,module in model.named_modules():
            if name.endswith(('input_layernorm','post_attention_layernorm','hidden_norm','norm','q_norm','k_norm')) and hasattr(module,'variance_epsilon'):
                pending_norm={}
                def norm_pre(m,args,key=name,pre=pending_norm):
                    if state['active']:pre[key]=[a.clone() if isinstance(a,torch.Tensor) else a for a in args]
                def hook(m,args,out,key=name,pre=pending_norm):
                    if not state['active']:return
                    args=pre[key]
                    state['current']['norms'].append(dict(name=key,input=cpu(args[0]),
                        residual=cpu(args[1]) if len(args)>1 else None,
                        output=cpu(out[0] if isinstance(out,tuple) else out),
                        residual_output=cpu(out[1]) if isinstance(out,tuple) else None,
                        weight=cpu(m.weight),eps=m.variance_epsilon))
                module.register_forward_pre_hook(norm_pre);module.register_forward_hook(hook)
            if hasattr(module,'_convolve'):
                original=module._convolve
                def convolve(x,delta,side,output_scale=1.,fn=original,m=module,key=name):
                    result=fn(x,delta,side,output_scale)
                    if state['active']:
                        state['current']['convs'].append(dict(name=key,input=cpu(x),delta=cpu(delta),
                            base=cpu(m.base_kernel[side]),side=side,block_size=m.block_size,
                            group_size=m.group_size,output_scale=output_scale,output=cpu(result)))
                    return result
                module._convolve=convolve
            if name.endswith('rotary_emb'):
                def rope_hook(m,args,out,key=name):
                    if state['active']:
                        state['current']['ropes'].append(dict(name=key,positions=args[0].detach().cpu(),
                            q_before=cpu(args[1]),k_before=cpu(args[2]),q_after=cpu(out[0]),k_after=cpu(out[1]),
                            head_size=m.head_size,neox=m.is_neox_style,base=m.base,
                            cos_sin=cpu(m.cos_sin_cache.index_select(0,args[0].flatten())),
                            cache_dtype=str(m.cos_sin_cache.dtype)))
                # RoPE implementations may mutate inputs; preserve them before
                # the forward, then pair with the resulting output.
                pending={}
                def rope_pre(m,args,key=name,pre=pending):
                    if state['active']:pre[key]=[a.clone() for a in args]
                def rope_post(m,args,out,key=name,pre=pending):
                    if state['active']:rope_hook(m,pre[key],out,key)
                module.register_forward_pre_hook(rope_pre);module.register_forward_hook(rope_post)
        original_score=s._score_candidates
        def score(ids,unary,hidden):
            result=original_score(ids,unary,hidden)
            if state['active']:
                selector=model.candidate_selector
                anchor=s.input_buffers.input_ids[:s.num_query_per_req:s.num_query_per_req]
                predecessor=torch.cat((anchor[:,None,None].expand(-1,1,16),ids[:,:-1]),1)
                projection=selector.hidden_projection(hidden)
                state['current']['selector']=dict(candidate_ids=ids.detach().cpu(),unary=cpu(unary),
                    hidden=cpu(hidden),projection=cpu(projection),projection_weight=cpu(selector.hidden_projection.weight),
                    anchor_ids=anchor.detach().cpu(),predecessor_ids=predecessor.detach().cpu(),
                    predecessor_rows=cpu(selector.predecessor_codebook[predecessor]),
                    successor_rows=cpu(selector.successor_codebook[ids]),scores=cpu(result))
            return result
        s._score_candidates=score
        import vllm.v1.attention.backends.flash_attn as fa
        original_fa=fa.flash_attn_varlen_func
        def attention(*args,**kwargs):
            value=original_fa(*args,**kwargs)
            q=kwargs.get('q')
            if state['active'] and q is not None and q.shape[-1]==128 and q.shape[0]==8 and not state['current'].get('attention'):
                table=kwargs['block_table'];length=int(kwargs['seqused_k'][0].cpu());k=kwargs['k'];v=kwargs['v']
                page=k.shape[1];start=max(0,(length-8-2047)//page)*page
                pages=table[0,start//page:(length+page-1)//page].long()
                kd=k[pages].reshape(-1,8,128)[:length-start].float()
                vd=v[pages].reshape(-1,8,128)[:length-start].float()
                ks=kwargs.get('k_descale');vs=kwargs.get('v_descale');qs=kwargs.get('q_descale')
                if ks is not None:kd*=ks[0].view(1,8,1)
                if vs is not None:vd*=vs[0].view(1,8,1)
                assert qs is None,'Capture requires nonquantized query'
                out=kwargs.get('out',value)
                state['current']['attention']=dict(q=cpu(q),k=cpu(kd),v=cpu(vd),out=cpu(out),
                    logical_start=start,length=length,page=page,block_table=table.detach().cpu(),
                    causal=kwargs['causal'],window=kwargs.get('window_size'),scale=kwargs['softmax_scale'],
                    stored_k_dtype=str(k.dtype),stored_v_dtype=str(v.dtype),
                    k_scale=cpu(ks),v_scale=cpu(vs))
            return value
        fa.flash_attn_varlen_func=attention
        original_propose=s.propose;sig=inspect.signature(original_propose)
        def propose(*args,**kwargs):
            bound=sig.bind(*args,**kwargs).arguments
            # The proposer also runs during chunked prompt prefill. Selected
            # rounds must refer to the eight-row decode geometry; its FC
            # inputs cannot be compared with an eight-row feature slice from
            # a4096-row prefill. This frozen32722-token fixture has no tiny
            # prefill tail (all prompt chunks exceed8 rows).
            if bound['input_batch'].num_tokens>s.num_query_per_req:
                state['active']=False
                return original_propose(*args,**kwargs)
            state['round']+=1;state['active']=state['round'] in state['selected']
            if state['active']:
                assert bound['input_batch'].num_reqs==1
                actual_features=bound['aux_hidden_states'];assert len(actual_features)==5
                state['current']=dict(round=state['round'],target_query_rows=bound['input_batch'].num_tokens,
                    norms=[],convs=[],ropes=[],cache_writes=[],
                    features=[dict(layer=layer-1,hidden=cpu(features[layer][0]),residual=cpu(features[layer][1]),
                        post=cpu(features[layer][2]),passed=cpu(actual_features[i][-8:]))
                        for i,layer in enumerate(core.aux_hidden_state_layers)],
                    num_rejected=cpu(bound['num_rejected']),num_sampled=cpu(bound['num_sampled']),
                    previous_context_positions=s.context_positions[:8].detach().cpu())
            result=original_propose(*args,**kwargs)
            if state['active']:
                state['current'].update(proposed_tokens=s.draft_tokens[:1].detach().cpu(),
                    sample_pos=s.sample_pos[:7].detach().cpu(),input_ids=s.input_buffers.input_ids[:8].detach().cpu(),
                    sample_indices=s.sample_indices[:7].detach().cpu(),
                    query_positions=s.input_buffers.positions[:8].detach().cpu(),
                    context_positions=s.context_positions[:8].detach().cpu(),
                    context_slots=s._context_slot_mappings[:,:8].detach().cpu(),
                    realized_scores=cpu(s.candidate_sampler.scores[:1]))
                path=folder/f'round-{state["round"]:04}.pt';torch.save(state['current'],path)
                state['files'].append(str(path));state['active']=False;state['current']=None
            return result
        s.propose=propose;r._branch_capture=state
        return dict(status='INSTALLED_EAGER_DIAGNOSTIC_ONLY',selected=sorted(state['selected']))

    def finish_branch_capture(self):
        s=self.model_runner._branch_capture
        result=dict(rounds=s['round'],files=s['files'],scope='Actual eager branches; graph replay q and synthetic graph attention are separate receipts')
        return result
