"""Independent CPU arithmetic for actual eager branch and stored-KV captures."""
import argparse
import json
import math
from pathlib import Path
import torch

RTOL,ATOL=.01,.005
def check(actual,reference,kind):
    assert actual.shape==reference.shape,(kind,actual.shape,reference.shape)
    error=(actual.double()-reference.double()).abs()
    result=dict(kind=kind,max_abs=float(error.max()),rms=float(error.square().mean().sqrt()),
        rtol=RTOL,atol=ATOL,finite=bool(torch.isfinite(actual).all()),
        allclose=bool(torch.allclose(actual.double(),reference.double(),rtol=RTOL,atol=ATOL)))
    assert result['finite'] and result['allclose'],result
    return result

def replay(path):
    x=torch.load(path,map_location='cpu',weights_only=False);checks=[];ideal_rope_errors=[]
    assert [f['layer'] for f in x['features']]==[5,19,33,47,61]
    for f in x['features']:
        reference=(f['hidden']+(f['residual'] if f['residual'] is not None else 0)).half().float()
        assert torch.equal(reference,f['post']) and torch.equal(f['post'],f['passed'])
    fc=x['feature_fc'];assert torch.equal(fc['input'],torch.cat([f['passed'] for f in x['features']],-1))
    checks.append(check(fc['output'],fc['input'].double()@fc['weight'].double().t(),'feature_fc_selected_columns'))
    for n in x['norms']:
        value=n['input'].double()
        if n['residual'] is not None:value=value+n['residual'].double()
        # FP32 fused-add/RMSNorm may retain the sum before a FP16 residual store.
        reference=value*torch.rsqrt(value.square().mean(-1,keepdim=True)+n['eps'])*n['weight'].double()
        checks.append(check(n['output'],reference,'norm:'+n['name']))
    for c in x['convs']:
        inp=c['input'].float();base=c['base'];delta=c['delta'];out=torch.zeros_like(inp)
        channels=torch.arange(inp.shape[1]);groups=channels//c['group_size']
        positions=torch.arange(inp.shape[0])%c['block_size']
        for tap in range(base.shape[0]):
            valid=positions>=tap
            coeff=base[tap]+delta[:,tap,groups]
            out[valid]+=coeff[valid]*inp[torch.arange(inp.shape[0])[valid]-tap]
        checks.append(check(c['output'],(out*c['output_scale']).half().float(),'conv:'+c['name']))
    projection=x['context_projection'];value=projection['input'].double()
    normalized=(value*torch.rsqrt(value.square().mean(-1,keepdim=True)+projection['eps'])*projection['norm_weight'].double()).half().double()
    expected=normalized@projection['weight'].double().t()
    kv=torch.stack((projection['k'],projection['v']),2).permute(1,0,2,3,4).flatten(1)
    checks.append(check(kv[:,projection['columns']],expected,'context_norm_and_selected_kv_projection_columns'))
    assert x['cache_writes'],'No actual first-layer cache writes captured'
    for write in x['cache_writes']:
        valid=write['valid']
        for name in ['k','v']:
            value=write['key' if name=='k' else 'value'][valid].reshape(-1,8,128)
            expected=(value/write[name+'_scale']).to(torch.float8_e4m3fn).float()
            assert torch.equal(expected,write['stored_'+name]),'Actual FP8 cache write mismatch'
    for r in x['ropes']:
        assert r['neox'] and r['head_size']==128
        assert r['base']==10000000. and r['cache_dtype']=='torch.float16'
        frequency=10000000.**(-torch.arange(0,128,2).double()/128)
        angles=r['positions'].double()[:,None]*frequency[None]
        ideal_cache=torch.cat((angles.cos(),angles.sin()),-1)
        # Cache construction uses GPU FP32 frequencies/angles and then FP16
        # coefficients. Validate theta independently, then isolate the actual
        # rotation kernel against exactly those stored coefficients. Do not
        # hide the separate error against ideal FP64 rotations.
        checks.append(check(r['cos_sin'],ideal_cache,'rope_cache_theta:'+r['name']))
        for which in ['q','k']:
            inp=r[which+'_before'].double().reshape(len(angles),-1,128)
            first,second=inp.chunk(2,-1);cos=angles.cos()[:,None];sin=angles.sin()[:,None]
            ideal=torch.cat((first*cos-second*sin,second*cos+first*sin),-1).reshape(r[which+'_after'].shape)
            error=(r[which+'_after'].double()-ideal).abs()
            ideal_rope_errors.append(dict(name=r['name'],which=which,max_abs=float(error.max()),
                rms=float(error.square().mean().sqrt()),within_original_bounds=bool(torch.allclose(r[which+'_after'].double(),ideal,rtol=RTOL,atol=ATOL))))
            cos,sin=r['cos_sin'].double().chunk(2,-1);cos=cos[:,None];sin=sin[:,None]
            ref=torch.cat((first*cos-second*sin,second*cos+first*sin),-1).reshape(r[which+'_after'].shape)
            checks.append(check(r[which+'_after'],ref,'rope_kernel_same_stored_coefficients:'+r['name']+':'+which))
    s=x['selector']
    final=[n for n in x['norms'] if n['name']=='norm']
    assert len(final)==1,'Missing actual final draft norm capture'
    # sample_pos is an absolute predicted-token position, not a row index.
    # Older captures retain both absolute arrays, which independently recover
    # the unique query row without assuming a numeric offset.
    matches=x['sample_pos'][:,None]==x['query_positions'][None]
    assert matches.sum(-1).eq(1).all()
    indices=matches.long().argmax(-1)
    if 'sample_indices' in x:assert torch.equal(indices,x['sample_indices'])
    assert torch.equal(final[0]['output'][indices],s['hidden'].reshape(7,5120)), 'Selector hidden states differ from actual final norm rows'
    projection=s['hidden'].double()@s['projection_weight'].double().t()
    checks.append(check(s['projection'],projection,'selector:hidden_projection'))
    product=(s['predecessor_rows'].double()*s['projection'].double()[:,:,None]).half().double()
    edge=torch.einsum('blpr,blcr->blpc',product,s['successor_rows'].double())
    # The actual edge GEMM narrows to FP16 before unary addition. Preserve
    # that explicit ABI rounding, while computing the dot independently.
    ref=(edge.half().float()+s['unary'][:,:,None]).half().float()
    checks.append(check(s['scores'],ref,'selector:unary_and_edges'))
    previous=0
    for step in range(7):
        scores=s['scores'][0,step,previous]
        assert torch.equal(scores,x['realized_scores'][0,step])
        selected=int(scores.argmax());token=s['candidate_ids'][0,step,selected]
        assert token==x['proposed_tokens'][0,step]
        previous=selected
    a=x['attention'];assert a['causal'] is False and a['window'] in ([2047,2047],(2047,2047))
    q=a['q'].double().view(8,8,4,128).permute(1,2,0,3)
    k=a['k'].double().permute(1,2,0);v=a['v'].double().permute(1,0,2)
    scores=torch.matmul(q,k[:,None])*a['scale']
    qpos=a['length']-8+torch.arange(8);kpos=a['logical_start']+torch.arange(a['k'].shape[0])
    visible=(qpos[:,None]-kpos[None]).abs()<=2047
    scores.masked_fill_(~visible[None,None],-math.inf)
    out=torch.matmul(scores.softmax(-1),v[:,None]).permute(2,0,1,3).reshape(8,32,128)
    checks.append(check(a['out'],out,'actual_paged_fp8_vs_dense_same_stored_kv'))
    assert x['input_ids'][1:].eq(248070).all()
    return dict(round=x['round'],features_post_residual_exact=True,greedy_selector_walk_exact=True,
        checks=checks,ideal_fp64_rope_errors=ideal_rope_errors,
        query_positions=x['query_positions'].tolist(),context_slots=x['context_slots'].tolist(),
        limitations='Actual eager model path. Existing graph q captures and synthetic graph/paged gates are separate; not bit-identical branch captures inside a production graph.')

def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    paths=sorted(a.folder.glob('round-*.pt'));assert len(paths)==4
    result=dict(status='PASS_ACTUAL_EAGER_BRANCH_SELECTOR_ROPE_STORED_KV_REPLAY',rounds=[replay(p) for p in paths])
    (a.folder/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])
if __name__=='__main__':main()
