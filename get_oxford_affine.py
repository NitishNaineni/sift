#!/usr/bin/env python3
import os, urllib.request, tarfile, tempfile

BASE = 'https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/'
SEQS = ['graf', 'wall', 'bark', 'boat', 'leuven', 'ubc']
DEST = 'data/oxford_affine'
os.makedirs(DEST, exist_ok=True)

for name in SEQS:
    seq_dir = os.path.join(DEST, name)
    if os.path.exists(seq_dir) and os.listdir(seq_dir):
        print(f'⏭  {name} exists')
        continue
    
    print(f'⇩  {name}')
    tmp_path = tempfile.mktemp(suffix='.tar.gz')
    urllib.request.urlretrieve(f'{BASE}{name}.tar.gz', tmp_path)
    
    os.makedirs(seq_dir, exist_ok=True)
    with tarfile.open(tmp_path) as tf:
        tf.extractall(seq_dir, filter='data')
    os.remove(tmp_path)
    print(f'✓  {name}')

print(f'Done: {DEST}')