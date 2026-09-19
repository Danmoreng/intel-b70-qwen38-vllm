"""Frozen exact-token cold and warm requests, calibrated with production template."""
import hashlib
import importlib.util
import json
from pathlib import Path
from transformers import AutoTokenizer

spec=importlib.util.spec_from_file_location('generate','/scripts/generate-exact-prompts.py')
generate=importlib.util.module_from_spec(spec);spec.loader.exec_module(generate)
tokenizer=AutoTokenizer.from_pretrained('/model',local_files_only=True)
fixtures=[]
for target in (8192,16384,32768,65536):
    for rep in range(6):
        entropy='RUN'+hashlib.sha256(f'B70-pro-review-{target}-{rep}'.encode()).hexdigest()
        family=list(generate.FAMILIES)[rep%5]
        content=generate.exact_content(tokenizer,target,family,entropy)
        prefix=generate.exact_content(tokenizer,target-256,family,entropy)
        fixtures.append({'target':target,'rep':rep,'family':family,
            'messages':[{'role':'user','content':content}],
            'prefix_messages':[{'role':'user','content':prefix}],
            'prefix_target':target-256})
Path('/out/fixtures.json').write_text(json.dumps(fixtures,indent=2)+'\n')
print('Wrote',len(fixtures),'calibrated full/prefix pairs.',flush=True)
