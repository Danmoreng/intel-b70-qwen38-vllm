#!/usr/bin/env python3
"""Exercise the updated grammar dependency through two real API contracts."""
import argparse
import json
from pathlib import Path
import urllib.request
import urllib.error

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base', required=True)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
records = {'status': 'RUNNING', 'base': a.base, 'responses': {}}
a.out.parent.mkdir(parents=True, exist_ok=True)
assert not a.out.exists()


def ask(label, prompt, **extra):
    body = {'model': 'Qwen3.8-27B', 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'max_tokens': 128, 'chat_template_kwargs': {'enable_thinking': False}, **extra}
    request = urllib.request.Request(a.base + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    records['responses'][label] = {'request': body, 'response': result}
    a.out.write_text(json.dumps(records, indent=2) + '\n')
    return result['choices'][0]['message']


try:
    schema = {'type': 'object', 'properties': {'city': {'type': 'string', 'enum': ['Paris']},
              'answer': {'type': 'integer', 'enum': [42]}}, 'required': ['city', 'answer'],
              'additionalProperties': False}
    message = ask('json-schema', 'Return city Paris and answer42 as JSON.', response_format={
        'type': 'json_schema', 'json_schema': {'name': 'result', 'strict': True, 'schema': schema}})
    assert json.loads(message['content']) == {'city': 'Paris', 'answer': 42}
    tool = {'type': 'function', 'function': {'name': 'add', 'description': 'Add two integers.',
        'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                       'required': ['a', 'b'], 'additionalProperties': False}}}
    message = ask('forced-tool', 'Call add with a=13 and b=29.', tools=[tool],
                  tool_choice={'type': 'function', 'function': {'name': 'add'}})
    call = message['tool_calls'][0]['function']
    assert call['name'] == 'add' and json.loads(call['arguments']) == {'a': 13, 'b': 29}
    records['status'] = 'PASS_STRUCTURED_JSON_AND_FORCED_TOOL'
except BaseException as error:
    records.update(status='FAILED', error=repr(error))
    if isinstance(error, urllib.error.HTTPError):
        records['http_error_body'] = error.read().decode()
    raise
finally:
    a.out.write_text(json.dumps(records, indent=2) + '\n')
