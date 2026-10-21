"""試験7 ピンセットで皿から皿へ移す（TRACK=1・乱れなし・seed 2026・±2cm）。chop_transfer.py を写して import を tweezers_pick にした。
cd ~/Developer/gmo-hand/lerobot && TRANSFER=1 BALL_CONDIM=6 T_HOLD=3 BALL_MU=0.6 BEAN_D_MM=.. .venv/bin/python <このファイル> <名前> [回数] [動画 1/0]"""
import os, sys, contextlib, io
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parent   # このファイルのフォルダ（ほかの sim/*.py と同じ場所）
sys.path.insert(0, str(W)); os.chdir(W)
assert os.environ.get("TRANSFER") == "1", "TRANSFER=1 を付けて走らせる"
import tweezers_pick as cp
tag = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
video = (sys.argv[3] == "1") if len(sys.argv) > 3 else True
d = os.environ["BEAN_D_MM"]
rng = np.random.default_rng(2026)
base = cp.BEAN.copy()
res = []
for k in range(n):
    off = rng.uniform(-0.02, 0.02, 2)
    v = video and k == 0
    cp.OUT = W / f"tweezers_transfer_{d}mm_{tag}.mp4"
    with contextlib.redirect_stdout(io.StringIO()) if not os.environ.get("DEBUG") else contextlib.nullcontext():
        r = cp.main(bean=base + [*off, 0], video=v, track=True)
    res.append(r)
    lost = "離れず" if r["lost_t"] is None else f"ピンセットから離れた {r['lost_t']:.2f}秒 @({r['lost_xyz'][0]*1000:.0f},{r['lost_xyz'][1]*1000:.0f},{(r['lost_xyz'][2]-cp.PLATE_TOP-cp.BEAN_R)*1000:+.0f})mm"
    fx = r["final_xyz"]
    print(f"{tag} d={d} #{k+1} off=({off[0]*100:+.1f},{off[1]*100:+.1f})cm 持ち上げ{'成功' if r['success'] else '失敗'} "
          f"{'移せた' if r['transferred'] else '移せず'} 皿2中心から{r['final_xy_err']:.1f}mm (a,b,c)={r['flags']} | "
          f"3.8-4.8秒の最大{r['lift_max']*1000:.1f}mm 運ぶ間の最小{r['carry_min']*1000:.1f}mm 全体の最大{r['max_h']*1000:.1f}mm "
          f"{lost} 最後({fx[0]*1000:.0f},{fx[1]*1000:.0f},{(fx[2]-cp.PLATE_TOP-cp.BEAN_R)*1000:+.1f})mm 最後の速さ{r['final_speed']:.1f}mm/s 腕のずれ最大{r['worst']:.1f}mm 運ぶ終わり{r['t1']:.1f}秒", flush=True)
print(f"SUMMARY {tag} d={d} 移せた {sum(r['transferred'] for r in res)}/{n} 持ち上げ {sum(r['success'] for r in res)}/{n} 距離 "
      + ",".join(f"{r['final_xy_err']:.0f}" for r in res))
