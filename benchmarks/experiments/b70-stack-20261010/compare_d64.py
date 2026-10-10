"""CPU-only comparison of actual same-prefix operands; no synthetic outputs."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dflash2-pro-diagnostics-v031'))
from rng_reference import acceptance_uniform,known_answers

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dist(x):return torch.log_softmax(x.double(),dim=-1)
def compare(x,y,scores=False):
    assert x.shape==y.shape
    a,b=dist(x),dist(y);p,q=a.exp(),b.exp();mask=p>0
    kl=float((p[mask]*(a[mask]-b[mask])).sum())
    tv=float((p-q).abs().sum()/2)
    finite=torch.isfinite(x)&torch.isfinite(y)
    middle=torch.logaddexp(a,b)-math.log(2);mask_q=q>0
    js=float(.5*((p[mask]*(a[mask]-middle[mask])).sum()+(q[mask_q]*(b[mask_q]-middle[mask_q])).sum()))
    return dict(bitidentical=torch.equal(x,y),kl_x_to_y=kl if math.isfinite(kl) else 'infinite_due_to_removed_support',js_divergence=js,total_variation=tv,
        top1_equal=int(x.argmax())==int(y.argmax()),x_top1=int(x.argmax()),y_top1=int(y.argmax()),
        max_finite_abs=float((x[finite]-y[finite]).abs().max()) if finite.any() else None,
        support_mask_equal=bool(torch.equal(torch.isfinite(x),torch.isfinite(y))))
def oracle(d):
    rows=[]
    for h,s,processed in zip(d['heads'],d['samples'],d['processed']):
        if h['has_prefill']:continue
        assert processed.shape[0]==h['input_ids'].numel()
        seed=int(s['seed'].flatten()[0]);temperature=float(s['temperature'].flatten()[0]);assert temperature==1
        expected=0;details=[];verifying=True
        for i,token in enumerate(h['input_ids'].flatten()[1:].tolist()):
            pos=int(h['positions'][i]);u=acceptance_uniform(seed,pos)
            prob=float(dist(processed[i])[token].exp()) if token>=0 else 0
            accepted=token>=0 and prob>u
            details.append(dict(step=i,probability=prob,uniform=u,prefix_verifier_visits_this_step=verifying,ratio_test_would_pass=accepted,accepted_in_prefix=verifying and accepted,position=pos,
                absolute_decision_margin=abs(prob-u),conditioning='Own-version preceding draft branch'))
            if verifying and accepted:expected+=1
            else:verifying=False
        native=int(s['num_sampled'].flatten()[0])-1
        assert native==expected,('Independent rejection oracle mismatch',native,expected,details)
        rows.append(dict(native_accepted=native,oracle_accepted=expected,decisions=details))
    assert rows
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    campaign=json.loads((a.root/'campaign.json').read_text());assert campaign['status']=='COMPLETE_BOUNDED_D64_CAPTURE_COMPARISON_PENDING'
    roles=list(campaign['arms']);data={};receipts=[]
    for role in roles:
        for pos in campaign['positions']:
            for n in range(2):
                path=a.root/role/'d64'/f'prefix-{pos}-repeat-{n}'/'capture.pt'
                d=torch.load(path,map_location='cpu',weights_only=False);data[role,pos,n]=d
                receipts.append(dict(role=role,position=pos,repeat=n,capture_sha256=sha(path),rejection=oracle(d)))
    repeatability=[];pairs=[]
    for role in roles:
        for pos in campaign['positions']:
            x,y=data[role,pos,0],data[role,pos,1]
            exact=all(torch.equal(h['logits'],j['logits']) for h,j in zip(x['heads'],y['heads']))
            repeatability.append(dict(role=role,position=pos,heads_bitidentical=exact,
                proposals_equal=all(torch.equal(h['tokens'],j['tokens']) for h,j in zip(x['proposals'],y['proposals'])),
                samples_equal=all(torch.equal(h['actual_sampled'],j['actual_sampled']) for h,j in zip(x['samples'],y['samples']))))
    for other in [r for r in roles if r!='qualified']:
        for pos in campaign['positions']:
            x,y=data['qualified',pos,0],data[other,pos,0]
            rows=[]
            # Prefill predicts the shared anchor. First verify row predicts
            # the next token after that shared anchor. Further rows compare
            # only while all preceding actual inputs are identical.
            for head in [0,1]:
                hx,hy=x['heads'][head],y['heads'][head]
                count=min(hx['input_ids'].numel(),hy['input_ids'].numel())
                for i in range(count):
                    if not torch.equal(hx['positions'][:i+1],hy['positions'][:i+1]) or not torch.equal(hx['input_ids'][:i+1],hy['input_ids'][:i+1]):break
                    rows.append(dict(head=head,row=i,position=int(hx['positions'][i]),
                        raw=compare(hx['logits'][i],hy['logits'][i]),
                        processed=compare(x['processed'][head][i],y['processed'][head][i],True)))
            assert any(r['head']==0 for r in rows) and any(r['head']==1 for r in rows)
            px,py=x['proposals'][0]['tokens'].flatten(),y['proposals'][0]['tokens'].flatten()
            equal=px==py;prefix=0
            for same in equal.tolist():
                if not same:break
                prefix+=1
            pairs.append(dict(other=other,position=pos,same_conditioning_rows=rows,
                first_proposal_equal_steps=equal.tolist(),first_proposal_common_prefix_length=prefix,
                note='Proposal step zero shares prefix and anchor. Any later step after a differing earlier proposal has different conditioning; it is not a numerical same-prefix comparison.'))
    report=dict(status='PASS_ACTUAL_REJECTION_ORACLE_BOUNDED_D64_REVIEW_REQUIRED',known_answer_vectors=known_answers(),
        source_sha256={'compare_d64.py':sha(Path(__file__)), 'rng_reference.py':sha(Path(__file__).resolve().parents[1]/'dflash2-pro-diagnostics-v031/rng_reference.py')},source_capture_receipts=receipts,repeatability=repeatability,comparisons=pairs,
        limits='Two frozen positions, two own-state reconstructions each. No throughput result, exhaustive quality proof, or claim of cross-version hidden-state identity. Raw arrays remain private. Different draft branches are excluded from same-prefix distribution comparisons.')
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=report['status'],repeatability=repeatability)),flush=True)
if __name__=='__main__':main()
