#!/usr/bin/env python3
"""Five real-weight draft layers on captured target states, without a KV pool.

Calls the patched vLLM decoder forward and convolution. The standalone attention
uses explicit GQA/SWA with FP32 softmax, not the paged serving attention backend.
The reference retains residuals in FP32; linear/attention/MLP weights and outputs
stay FP16. This isolates residual representation error from checkpoint conversion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from safetensors import safe_open
from vllm.config import VllmConfig, set_current_vllm_config
SCALE=1/16
EPS=1e-6
TRACE=[]

def file_sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def norm(x,w,eps):
    xf=x.float()
    return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+eps)*w.float()).to(torch.float16)

class ReferenceNorm(nn.Module):
    def __init__(self,w):super().__init__();self.w=w
    def forward(self,x,residual=None):
        if residual is None:return norm(x,self.w,EPS)
        summed=x.float()+residual.float()
        return norm(summed,self.w,EPS),summed

def reference_conv(x,delta,base,block):
    blocks=x.float().reshape(-1,320,16)
    coefficients=base.float().view(1,2,320,16)+delta.float().unsqueeze(-1)
    y=coefficients[:,0]*blocks
    previous=F.pad(blocks[:-1],(0,0,0,0,1,0))
    active=(torch.arange(x.shape[0],device=x.device)%block>=1).view(-1,1,1)
    return (y+coefficients[:,1]*previous*active).flatten(-2)

class Conv(nn.Module):
    def __init__(self,w,prefix,block,reference=False):
        super().__init__();self.base_kernel=w[prefix+'.base_kernel'];self.weight=w[prefix+'.kernel_projection.weight']
        self.block_size=block;self.num_groups=320;self.group_size=16;self.taps=2;self.reference=reference
    def kernel_projection(self,x):return F.linear(x,self.weight)
    def _convolve(self,x,delta,side,output_scale=1.):
        if self.reference and side==1:return reference_conv(x,delta,self.base_kernel[side],self.block_size)
        from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv
        return _grouped_conv(x,delta,self.base_kernel[side],self.block_size,320,16,2,output_scale)
    def prepare(self,x):
        from vllm.model_executor.models.qwen3_dflash2 import DFlashGroupedConv
        return DFlashGroupedConv.prepare(self,x)
    def finish(self,x,c,output_scale=1.):
        from vllm.model_executor.models.qwen3_dflash2 import DFlashGroupedConv
        y=DFlashGroupedConv.finish(self,x,c,output_scale)
        TRACE.append(dict(stage='conv-finish',finite=bool(torch.isfinite(y).all()),absmax=float(y.abs().max()),dtype=str(y.dtype)))
        return y

class MLP(nn.Module):
    def __init__(self,w,prefix):super().__init__();self.w=w;self.prefix=prefix
    def forward(self,x):
        return F.linear(F.silu(F.linear(x,self.w[self.prefix+'.gate_proj.weight']))*F.linear(x,self.w[self.prefix+'.up_proj.weight']),self.w[self.prefix+'.down_proj.weight'])

def rope(x,positions):
    freq=1./(10000000.**(torch.arange(0,128,2,device=x.device,dtype=torch.float32)/128))
    angles=positions.float()[:,None]*freq[None,:]
    cos,sin=angles.cos().half()[:,None,:],angles.sin().half()[:,None,:]
    first,last=x[...,:64],x[...,64:]
    return torch.cat([first*cos-last*sin,last*cos+first*sin],dim=-1)

class Attention(nn.Module):
    def __init__(self,w,prefix,context,context_positions):
        super().__init__();self.w=w;self.prefix=prefix;self.ctxpos=context_positions
        self.ctxk=rope(norm(F.linear(context,w[prefix+'.k_proj.weight']).view(-1,8,128),w[prefix+'.k_norm.weight'],EPS),context_positions)
        self.ctxv=F.linear(context,w[prefix+'.v_proj.weight']).view(-1,8,128)
    def forward(self,positions,hidden_states):
        x=hidden_states;w=self.w;p=self.prefix
        q=rope(norm(F.linear(x,w[p+'.q_proj.weight']).view(-1,32,128),w[p+'.q_norm.weight'],EPS),positions)
        k=rope(norm(F.linear(x,w[p+'.k_proj.weight']).view(-1,8,128),w[p+'.k_norm.weight'],EPS),positions)
        v=F.linear(x,w[p+'.v_proj.weight']).view(-1,8,128)
        keys=torch.cat([self.ctxk,k]).repeat_interleave(4,dim=1).transpose(0,1)
        values=torch.cat([self.ctxv,v]).repeat_interleave(4,dim=1).transpose(0,1)
        scores=(q.transpose(0,1).float()@keys.transpose(-1,-2).float())*(128**-.5)
        kvpos=torch.cat([self.ctxpos,positions])
        mask=(kvpos[None,:]>positions[:,None]-2048)&(kvpos[None,:]<positions[:,None]+2048)
        scores=scores.masked_fill(~mask[None,:,:],-float('inf'))
        probs=scores.softmax(-1)
        output=(probs@values.float()).half().transpose(0,1).reshape(-1,4096)
        return F.linear(output,w[p+'.o_proj.weight'])

def layer(weights,idx,block,mode,context,ctxpos):
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3DecoderLayer
    from vllm.model_executor.layers.layernorm import RMSNorm
    obj=DFlash2Qwen3DecoderLayer.__new__(DFlash2Qwen3DecoderLayer);nn.Module.__init__(obj)
    obj.residual_scale=SCALE if mode=='patched' else 1.
    p=f'layers.{idx}'
    for name in ['input_layernorm','post_attention_layernorm']:
        weight=weights[p+'.'+name+'.weight']
        if mode=='reference':module=ReferenceNorm(weight)
        else:
            module=RMSNorm(5120,eps=EPS*obj.residual_scale**2,dtype=torch.float16).to(weight.device)
            module.weight=nn.Parameter(weight,requires_grad=False)
        setattr(obj,name,module)
    obj.attention_conv=Conv(weights,p+'.attention_conv',block,mode=='reference')
    obj.mlp_conv=Conv(weights,p+'.mlp_conv',block,mode=='reference')
    obj.self_attn=Attention(weights,p+'.self_attn',context,ctxpos)
    obj.mlp=MLP(weights,p+'.mlp')
    return obj

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    checkpoint=Path('/draft/model.safetensors')
    assert file_sha(checkpoint)=='67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c'
    weights={}
    with safe_open(checkpoint,framework='pt',device='cpu') as f:
        for key in f.keys():
            # Checkpoint conversion is CPU-only and happens once before device upload.
            half=f.get_tensor(key).to(torch.float16)
            assert torch.isfinite(half).all(), key
            weights[key]=half.to(a.device)
    assert all(x.dtype==torch.float16 for x in weights.values())
    report=dict(status='RUNNING',scope='real-weight numerical replay; no paged KV/serving/acceptance qualification',checkpoint_sha256=file_sha(checkpoint),cases=[])
    config=VllmConfig();config.compilation_config.mode=0;config.kernel_config.set_platform_defaults(config)
    with set_current_vllm_config(config), config.kernel_config.ir_op_priority.set_priority(), torch.no_grad():
        for case in sorted(Path('/results/target-states').glob('case-*')):
            captures=[];row_masks=None
            for idx in [5,19,33,47,61]:
                chunks=[torch.load(f,weights_only=True) for f in sorted(case.glob(f'layer-{idx}-chunk-*.pt'))]
                hidden=torch.cat([c['hidden'] for c in chunks]);pos=torch.cat([c['positions'] for c in chunks])
                captures.append((hidden,pos))
                if row_masks is None:row_masks=[c['valid_rows'] for c in chunks]
            positions=captures[0][1]
            assert all(torch.equal(pos,positions) for _,pos in captures)
            embedding_files=sorted(case.glob('prompt-embeddings*.pt'))
            # The shared embedding module is called first by the target, then
            # by MTP with the teacher-forced next-token shift. Validate every pair.
            assert len(embedding_files)==2*len(row_masks)
            inputs=[torch.load(f,weights_only=True) for f in embedding_files]
            for target,mtp in zip(inputs[::2],inputs[1::2]):
                assert target['ids'].shape==mtp['ids'].shape
                assert torch.equal(target['ids'][1:],mtp['ids'][:-1])
            embeddings=torch.cat([c['embeddings'][valid] for c,valid in zip(inputs[::2],row_masks)]).to(a.device)
            assert torch.equal(positions,torch.arange(len(positions))), 'unexpected prefix/chunk order'

            assert len(embeddings)==len(positions)
            mask=torch.load(case/'mask-embedding.pt',weights_only=True).to(a.device)
            auxiliary=torch.cat([c[0] for c in captures],dim=-1).to(a.device)
            context=norm(F.linear(auxiliary,weights['fc.weight']),weights['hidden_norm.weight'],EPS)
            assert torch.isfinite(context).all()
            for anchor in [min(511,len(positions)-1),len(positions)-1]:
                lo=max(0,anchor-2047)
                ctx=context[lo:anchor+1]
                ctxpos=positions[lo:anchor+1].to(a.device)
                for block in [4,6,8]:
                    querypos=torch.arange(anchor+1,anchor+1+block,device=a.device)
                    bonus=(torch.load(case/'bonus-embedding.pt',weights_only=True).to(a.device) if anchor==len(positions)-1 else embeddings[anchor+1:anchor+2])
                    query=torch.cat([bonus,mask.expand(block-1,-1)],dim=0)
                    outputs={};traces={}
                    for mode in ['old','patched','reference']:
                        global TRACE
                        TRACE=[];x=query.clone();residual=None
                        for i in range(5):
                            obj=layer(weights,i,block,mode,ctx,ctxpos)
                            x,residual=obj(querypos,x,residual)
                            TRACE.append(dict(stage='layer-output',layer=i,hidden_finite=bool(torch.isfinite(x).all()),residual_finite=bool(torch.isfinite(residual).all()),residual_absmax=float(residual.abs().max()),dtype=str(residual.dtype)))
                            del obj
                        if mode=='reference':y=norm(x.float()+residual.float(),weights['norm.weight'],EPS)
                        else:
                            from vllm.model_executor.layers.layernorm import RMSNorm
                            final_norm=RMSNorm(5120,eps=EPS*(SCALE**2 if mode=='patched' else 1),dtype=torch.float16).to(a.device)
                            final_norm.weight=nn.Parameter(weights['norm.weight'],requires_grad=False)
                            y,_=final_norm(x,residual)

                        outputs[mode]=y;traces[mode]=TRACE
                    expected,actual=outputs['reference'].float(),outputs['patched'].float()
                    assert torch.isfinite(expected).all() and torch.isfinite(actual).all(), (case,anchor,block)
                    relative_l2=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected))
                    cosine=float(F.cosine_similarity(actual,expected,dim=-1).min())
                    assert relative_l2<.02 and cosine>.999, (relative_l2,cosine)
                    item=dict(case=case.name,anchor=anchor,block_size=block,context_tokens=len(ctx),old_finite=bool(torch.isfinite(outputs['old']).all()),patched_finite=True,relative_l2=relative_l2,min_cosine=cosine,traces=traces)
                    report['cases'].append(item);a.out.write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps({k:v for k,v in item.items() if k!='traces'}),flush=True)
    assert any(not c['old_finite'] for c in report['cases']), 'real-weight overflow was not reproduced'
    report['status']='PASS_REAL_WEIGHT_RESIDUAL_FIX_NOT_SERVING_QUALIFICATION'
    a.out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
