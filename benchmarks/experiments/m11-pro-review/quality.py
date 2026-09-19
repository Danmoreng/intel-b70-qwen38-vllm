"""Frozen semantics clarification; existing assertions are unchanged."""
from pathlib import Path
from common import load,REPO

quality=load('original_quality',REPO/'scripts/check-target-head-quality.py')
name,task,assertions=quality.TASKS[0]
quality.TASKS[0]=(name,task+' Precisely: merge when next_start <= current_end. '
    'Do NOT merge merely adjacent integers: [(1,10),(11,12)] must stay separate.',assertions)
# Review's production-sampling quality lane, distinct from greedy throughput.
post=quality.post
def production_post(root,endpoint,payload):
    if endpoint=='/v1/chat/completions':
        payload={**payload,'temperature':1,'top_p':.95,'top_k':20,'seed':190919}
    return post(root,endpoint,payload)
quality.post=production_post
if __name__=='__main__':quality.main()
