# -*- coding: utf-8 -*-
"""断点续拷：python resume_copy.py <src> <dst>，已存在部分以追加方式续传。"""
import os
import shutil
import sys
import time

src, dst = sys.argv[1], sys.argv[2]
src_size = os.path.getsize(src)
have = os.path.getsize(dst) if os.path.exists(dst) else 0
if have > src_size:
    os.remove(dst)
    have = 0
t0 = time.time()
with open(src, "rb") as f, open(dst, "ab") as g:
    f.seek(have)
    shutil.copyfileobj(f, g, 8 * 1024 * 1024)
    g.flush()
    os.fsync(g.fileno())
have = os.path.getsize(dst)
dt = time.time() - t0
print(f"{os.path.basename(dst)}: {have}/{src_size} bytes "
      f"({have / src_size * 100:.1f}%, +{(have - (0 if dt == 0 else 0))} "
      f"this run {dt:.0f}s)", flush=True)
if have == src_size:
    print("DONE", flush=True)
