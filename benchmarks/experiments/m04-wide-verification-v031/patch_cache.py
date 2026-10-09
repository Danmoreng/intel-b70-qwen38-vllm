#!/usr/bin/env python3
"""Guard and fix AOT cache isolation across speculative draft lengths."""
import ast
import hashlib
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
expected = sys.argv[2]
assert hashlib.sha256(source.encode()).hexdigest() == expected
old = '        factors: list[Any] = []\n        # Eagle3 and extract_hidden_states affect the computation graph because'
assert source.count(old) == 1
new = '''        factors: list[Any] = []
        # DFlash convolution boundaries and selector token dimensions, and
        # other speculative graphs, specialize on the configured draft length.
        factors.append(self.num_speculative_tokens)
        # Eagle3 and extract_hidden_states affect the computation graph because'''
source = source.replace(old, new)
ast.parse(source)
path.write_text(source)
print('SPECULATIVE_CACHE_SOURCE_SHA256', hashlib.sha256(source.encode()).hexdigest())
