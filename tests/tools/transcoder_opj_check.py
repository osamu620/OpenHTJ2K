#!/usr/bin/env python3
"""Decode a Part 1 stream and its transcoded HTJ2K with OpenJPEG (via Pillow) and compare.
usage: opjcheck.py <transcode-binary> <workdir> <files...>"""
import sys, os, subprocess, io
import numpy as np
from PIL import Image
tr, work, files = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(work, exist_ok=True)
def load(path):
    im = Image.open(path); im.load()
    return im.mode, np.asarray(im)
res = {}
for f in files:
    name = os.path.basename(f); out = os.path.join(work, name.rsplit('.',1)[0] + '_ht.' + ('jph' if f.endswith('.jp2') else 'j2c'))
    r = subprocess.run([tr, '-i', f, '-o', out], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{name:34s} TRANSCODE-REFUSED  {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[:80]}"); continue
    try: ma, a = load(f)
    except Exception as e: print(f"{name:34s} REF-UNREADABLE (Pillow/OpenJPEG on the input): {str(e)[:60]}"); continue
    try: mb, b = load(out)
    except Exception as e: print(f"{name:34s} *** OPENJPEG FAILS ON TRANSCODED: {str(e)[:70]}"); res[name]='fail'; continue
    if ma != mb or a.shape != b.shape: print(f"{name:34s} *** MISMATCH shape/mode {ma}{a.shape} vs {mb}{b.shape}"); res[name]='fail'; continue
    d = np.abs(a.astype(np.int64) - b.astype(np.int64))
    st = 'IDENTICAL' if d.max() == 0 else f'DIFF max {d.max()} count {(d>0).sum()}'
    print(f"{name:34s} {st}  ({ma}, {a.shape})"); res[name] = 'ok' if d.max()==0 else 'diff'
print(sum(v=='ok' for v in res.values()), 'identical;', sum(v!='ok' for v in res.values()), 'not')
