"""球を小さくする試験（TRACK=1・乱れなし・seed 2026・±2cm）。cd ~/Developer/gmo-hand/lerobot && BEAN_D_MM=.. [GRIP_KP=..] [DOWN_DZ_MM=..] .venv/bin/python <このファイル> <名前> [回数] [動画 1/0] [GRIP_BACK mm] [傾き 度]"""
import os, sys, contextlib, io
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parent   # このファイルのフォルダ（ほかの sim/*.py と同じ場所）
sys.path.insert(0, str(W)); os.chdir(W)
import chopsticks_pick as cp
tag = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
video = (sys.argv[3] == "1") if len(sys.argv) > 3 else True
if len(sys.argv) > 4 and sys.argv[4] != "-": cp.GRIP_BACK = float(sys.argv[4]) / 1000
if len(sys.argv) > 5 and sys.argv[5] != "-": cp.STICK_TILT_DEG = float(sys.argv[5])
d = os.environ["BEAN_D_MM"]
rng = np.random.default_rng(2026)
base = cp.BEAN.copy()
res = []
for k in range(n):
    off = rng.uniform(-0.02, 0.02, 2)
    v = video and k == 0
    cp.OUT = W / f"chopsticks_pick_{d}mm_track_small_{tag}.mp4"
    with contextlib.redirect_stdout(io.StringIO()) if not os.environ.get("DEBUG") else contextlib.nullcontext():
        r = cp.main(bean=base + [*off, 0], video=v, track=True)
    res.append(r)
    print(f"{tag} d={d} #{k+1} off=({off[0]*100:+.1f},{off[1]*100:+.1f})cm {'成功' if r['success'] else '失敗'} 最大{r['max_h']*1000:.1f}mm 最後{r['final_h']*1000:.1f}mm", flush=True)
print(f"SUMMARY {tag} d={d} {sum(r['success'] for r in res)}/{n} 最後の高さ " + ",".join(f"{r['final_h']*1000:.0f}" for r in res))
