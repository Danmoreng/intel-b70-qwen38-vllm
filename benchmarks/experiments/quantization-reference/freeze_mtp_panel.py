"""Freeze code/prose token inputs and the existing clean 103K coding history."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

from transformers import AutoTokenizer

PANEL_SHA='cbf1a71bbda470859f2c0786cb7134e260111d2072c4753a6732021dadae6171'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--coding-history',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Frozen output must be fresh')
    panel_bytes=args.panel.read_bytes();assert sha(panel_bytes)==PANEL_SHA
    panel=json.loads(panel_bytes);history_bytes=args.coding_history.read_bytes()
    historical=json.loads(gzip.decompress(history_bytes));history=copy.deepcopy(historical)
    # Match the API's tool-call preprocessing; the template expects mappings.
    for message in history['messages']:
        # vLLM chat_utils emits both fields; the checkpoint template reads
        # reasoning_content. Dropping that alias silently removes old thoughts.
        reasoning=message.get('reasoning')
        if reasoning is None: reasoning=message.get('reasoning_content')
        if reasoning is not None:
            message['reasoning']=reasoning
            message['reasoning_content']=reasoning
        for call in message.get('tool_calls',[]):
            function=call['function']
            if isinstance(function.get('arguments'),str):
                function['arguments']=json.loads(function['arguments'])
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    rendered=tokenizer.apply_chat_template(history['messages'],tools=history.get('tools'),
        tokenize=True,add_generation_prompt=True,**history['chat_template_kwargs'])
    clean=rendered['input_ids'] if isinstance(rendered,Mapping) else rendered
    assert len(clean)==102752,'Existing API 103K-history count differs from frozen rendering'
    windows=[]
    for context in (4096,32768,102752,131072):
        for domain in ('code','prose'):
            if domain=='code' and context==102752:
                ids=clean;recipe='Exact existing 103K coding history rendered with tools and preserved reasoning'
            else:
                indices=[i for i,w in enumerate(panel['windows']) if w['domain']==domain]
                material=[token for i in indices for token in panel['windows'][i]['ids']]
                instruction=('\nComplete the next function implementation and explain the reasoning.\n' if domain=='code'
                             else '\nContinue the article in coherent factual prose.\n')
                tail=tokenizer(instruction,add_special_tokens=False)['input_ids']
                needed=context-len(tail)
                ids=(material*((needed+len(material)-1)//len(material)))[:needed]+tail
                recipe='Repeat eight frozen '+domain+' source windows, followed by a fixed continuation instruction'
            assert len(ids)==context and all(isinstance(i,int) and 0<=i<248077 for i in ids)
            windows.append({'name':f'{domain}-{context}','domain':domain,'context_tokens':context,
                            'ids':ids,'ids_sha256':sha(json.dumps(ids,separators=(',',':')).encode()),
                            'recipe':recipe})
    result={'schema':1,'seed':20261001,'source_panel_sha256':PANEL_SHA,
            'coding_history_sha256':sha(history_bytes),'tokenizer_sha256':sha((args.model/'tokenizer.json').read_bytes()),
            'chat_template_sha256':sha(tokenizer.chat_template.encode()),
            'sampling':{'temperature':0,'seed':20261001,'max_tokens':512,'ignore_eos':True},
            'windows':windows,
            'scope':'Fixed-output performance inputs, not an adaptive coding task or code-quality score. C4 uses independent cache salts for identical frozen token inputs. Nonhistorical windows use repeated source material.'}
    data=(json.dumps(result,separators=(',',':'))+'\n').encode();encoded=gzip.compress(data,mtime=0)
    assert gzip.decompress(encoded)==data
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(encoded)
    manifest={k:v for k,v in result.items() if k!='windows'}
    manifest.update(builder_sha256=sha(Path(__file__).read_bytes()),panel_raw_sha256=sha(data),panel_gzip_sha256=sha(encoded),
                    windows=[{k:v for k,v in w.items() if k!='ids'} for w in windows])
    args.output.with_name('mtp-panel-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'raw_bytes':len(data),'compressed_bytes':len(encoded),
                      'windows':len(windows),'panel_sha256':sha(data)}),flush=True)


if __name__=='__main__':
    main()
