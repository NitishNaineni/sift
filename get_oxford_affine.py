#!/usr/bin/env python3
import os
import urllib.request
import tarfile
import tempfile
from PIL import Image
import glob

BASE = 'https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/'
SEQS = ['graf', 'wall', 'bark', 'boat', 'leuven', 'ubc']
DEST = 'data/oxford_affine'
os.makedirs(DEST, exist_ok=True)

for name in SEQS:
    seq_dir = os.path.join(DEST, name)
    if not os.path.exists(seq_dir) or not os.listdir(seq_dir):
        print(f'Downloading {name}...')
        tmp_path = tempfile.mktemp(suffix='.tar.gz')
        urllib.request.urlretrieve(f'{BASE}{name}.tar.gz', tmp_path)
        os.makedirs(seq_dir, exist_ok=True)
        with tarfile.open(tmp_path) as tf:
            tf.extractall(seq_dir, filter='data')
        os.remove(tmp_path)
    
    image_files = glob.glob(os.path.join(seq_dir, '*.ppm')) + glob.glob(os.path.join(seq_dir, '*.pgm'))
    if image_files:
        print(f'Converting {name} to PNG...')
        for img_file in image_files:
            png_file = img_file.replace('.ppm', '.png').replace('.pgm', '.png')
            if not os.path.exists(png_file):
                Image.open(img_file).save(png_file, 'PNG')
            os.remove(img_file)