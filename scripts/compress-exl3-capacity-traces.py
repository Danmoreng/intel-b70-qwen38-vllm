#!/usr/bin/env python3
"""Losslessly compress terminal capacity traces with SHA256/roundtrip receipts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve()
    state = json.loads((root / 'campaign.json').read_text())
    if state['status'] not in ('COMPLETE', 'FAILED'):
        raise RuntimeError('Only terminal campaigns may be compressed')
    receipt_file = root / 'trace-compression.json'
    previous = json.loads(receipt_file.read_text()) if receipt_file.exists() else {'files': []}
    files = {row['raw_file']: row for row in previous['files']}
    for case in root.iterdir():
        # Reused cases belong to the source campaign, not this receipt.
        if not case.is_dir() or case.is_symlink():
            continue
        for file in sorted((case / 'trace').glob('trace-*.jsonl')):
            data = file.read_bytes()
            encoded = gzip.compress(data, compresslevel=9, mtime=0)
            assert gzip.decompress(encoded) == data
            output = file.with_suffix(file.suffix + '.gz')
            if output.exists() and output.read_bytes() != encoded:
                raise RuntimeError('Existing compressed trace differs: ' + str(output))
            output.write_bytes(encoded)
            assert gzip.decompress(output.read_bytes()) == data
            relative = str(file.relative_to(root))
            files[relative] = {'raw_file': relative, 'compressed_file': str(output.relative_to(root)),
                               'raw_bytes': len(data), 'compressed_bytes': len(encoded),
                               'raw_sha256': sha(data), 'compressed_sha256': sha(encoded)}
            # Write proof before removing the equivalent uncompressed copy.
            receipt_file.write_text(json.dumps({'schema': 1, 'lossless_roundtrip_verified': True,
                                                'files': list(files.values())}, indent=2) + '\n')
            file.unlink()
    print(json.dumps({'campaign': str(root), 'files': len(files),
                      'raw_bytes': sum(row['raw_bytes'] for row in files.values()),
                      'compressed_bytes': sum(row['compressed_bytes'] for row in files.values())}))


if __name__ == '__main__':
    main()
