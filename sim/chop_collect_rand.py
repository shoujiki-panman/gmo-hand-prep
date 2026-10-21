"""試験10（10/16）：ばらつき入りのお手本を自動で集める。chop_collect.py を写して作った（元は変えない）。

- 課題＝皿から皿2へ移す（TRANSFER=1、終わり 8.0秒、T_HOLD 0）。成功＝transferred の回だけ保存
- 1回ごとに乱数で（§7）: 摩擦 BALL_MU 0.3〜1.0／直径 8〜10mm／質量 ×0.7〜1.3（BEAN_MASS_SCALE）／位置 ±2cm
  直径・質量・摩擦は chopsticks_pick の読み込み時に環境変数から決まるので、1回ごとに環境変数を書き換えて importlib.reload で読み直す
  読み直した値（cp.BALL_MU・cp.BEAN_R・cp.BEAN_MASS）と、組み立てたモデルの値（球の質量・摩擦）を CSV に並べて、実際に変わったことを確かめる
- 立て直し入り: 3割の回で、0.5〜1.8秒のどこかで球を 1.5cm（向きは乱数）ずらす。追従制御（track=True）が立て直す動きがお手本に残る
- 観測＝腕7＋箸1（球の位置は入れない）＋画像2台 128px、指示＝腕7＋箸1、20Hz
- 1本ごとの条件を chop_rand9_episodes.csv に書く

  cd ~/Developer/gmo-hand/lerobot && .venv/bin/python <このファイル> [集める本数]
"""

import contextlib
import csv
import importlib
import io
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ["TRANSFER"] = "1"
os.environ["T_HOLD"] = "0"
import chopsticks_pick as cp  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

REPO_ID = "tanuma/chop_rand9"
STATE_DIM = 8
ROOT = Path.home() / ".cache/huggingface/lerobot" / REPO_ID
CSV_PATH = HERE / "chop_rand9_episodes.csv"
TASK = "箸で豆を1粒つまんで皿2へ移す"
JOINT_NAMES = [j.removeprefix("right_").removesuffix("_joint") for j in cp.ARM_JOINTS]


def sample(rng):
    """1回分のばらつき。お手本・評価で共通に使う（評価は別の seed）"""
    return {"mu": rng.uniform(0.3, 1.0), "d_mm": rng.uniform(8.0, 10.0), "mass_scale": rng.uniform(0.7, 1.3),
            "offset": rng.uniform(-0.02, 0.02, 2)}


def apply(c):
    """環境変数を書き換えて chopsticks_pick を読み直す。読み直した値と、組み立てたモデルの値を返す"""
    global cp
    os.environ["BALL_MU"] = f"{c['mu']:.4f}"
    os.environ["BEAN_D_MM"] = f"{c['d_mm']:.4f}"
    os.environ["BEAN_MASS_SCALE"] = f"{c['mass_scale']:.4f}"
    cp = importlib.reload(cp)
    model, _ = cp.build(cp.Sticks())
    g = [i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body("bean0").id][0]
    return {"cp_mu": cp.BALL_MU, "cp_d_mm": cp.BEAN_R * 2000, "cp_mass_g": cp.BEAN_MASS * 1000,
            "model_mass_g": float(model.body("bean0").mass[0]) * 1000, "model_mu": float(model.geom_friction[g][0]),
            "model_r_mm": float(model.geom_size[g][0]) * 1000}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    if ROOT.exists():
        shutil.rmtree(ROOT)
    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": JOINT_NAMES + ["chop"]},
        "observation.images.side": {"dtype": "video", "shape": (cp.OBS_SIZE, cp.OBS_SIZE, 3),
                                    "names": ["height", "width", "channels"]},
        "observation.images.top": {"dtype": "video", "shape": (cp.OBS_SIZE, cp.OBS_SIZE, 3),
                                   "names": ["height", "width", "channels"]},
        "action": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES + ["chop"]},
    }
    ds = LeRobotDataset.create(REPO_ID, fps=cp.REC_HZ, features=features, root=ROOT,
                               robot_type="unitree_g1_brainco_chopsticks", use_videos=True)
    rng = np.random.default_rng(0)
    kept = tried = 0
    t_start = time.time()
    fields = ["episode", "try", "mu", "d_mm", "mass_scale", "dx_cm", "dy_cm", "perturb", "perturb_t", "perturb_dx_cm",
              "perturb_dy_cm", "cp_mu", "cp_d_mm", "cp_mass_g", "model_mu", "model_r_mm", "model_mass_g",
              "lift_max_mm", "final_xy_err_mm", "frames", "sec"]
    fcsv = open(CSV_PATH, "w", newline="")
    w = csv.DictWriter(fcsv, fieldnames=fields)
    w.writeheader()
    while kept < n and tried < n * 2:
        tried += 1
        t0 = time.time()
        c = sample(rng)
        do_perturb = rng.uniform() < 0.3
        pt, ang = rng.uniform(0.5, 1.8), rng.uniform(0, 2 * np.pi)
        perturb = (pt, 0.015 * np.cos(ang), 0.015 * np.sin(ang)) if do_perturb else None
        got = apply(c)
        frames = []

        def hook(ob, action):
            frames.append({"observation.state": ob["state"][:STATE_DIM],
                           "observation.images.side": ob["images"]["side"],
                           "observation.images.top": ob["images"]["top"],
                           "action": action, "task": TASK})

        with contextlib.redirect_stdout(io.StringIO()):
            r = cp.main(bean=cp.BEAN + [*c["offset"], 0], hook=hook, video=False, perturb=perturb, track=True)
        desc = (f"μ{c['mu']:.2f} 直径{c['d_mm']:.1f}mm 質量×{c['mass_scale']:.2f} ずらし {np.round(c['offset'] * 100, 1)}cm"
                f"{f' 乱れ {pt:.2f}秒' if do_perturb else ''}")
        if not r["transferred"]:
            print(f"{tried}回目: 移せず→保存しない（{desc}、持ち上げ {r['lift_max'] * 1000:.0f}mm、皿2中心から {r['final_xy_err']:.0f}mm）", flush=True)
            continue
        for f in frames:
            ds.add_frame(f)
        ds.save_episode(parallel_encoding=False)   # 並列だと描画中の親プロセスと合わず子プロセスが落ちた（10/14）
        kept += 1
        w.writerow({"episode": kept - 1, "try": tried, "mu": round(c["mu"], 4), "d_mm": round(c["d_mm"], 3),
                    "mass_scale": round(c["mass_scale"], 4), "dx_cm": round(c["offset"][0] * 100, 3),
                    "dy_cm": round(c["offset"][1] * 100, 3), "perturb": int(do_perturb),
                    "perturb_t": round(pt, 3) if do_perturb else "", "perturb_dx_cm": round(perturb[1] * 100, 3) if do_perturb else "",
                    "perturb_dy_cm": round(perturb[2] * 100, 3) if do_perturb else "",
                    **{k: round(v, 5) for k, v in got.items()}, "lift_max_mm": round(r["lift_max"] * 1000, 1),
                    "final_xy_err_mm": round(r["final_xy_err"], 1), "frames": len(frames), "sec": round(time.time() - t0, 1)})
        fcsv.flush()
        print(f"{tried}回目: 成功 → お手本 {kept}/{n}（{len(frames)}コマ、{desc}、{time.time() - t0:.1f}秒）", flush=True)
    ds.finalize()
    fcsv.close()
    print(f"保存: {ROOT}  お手本 {kept}本（試した回数 {tried}）  所要 {(time.time() - t_start) / 60:.1f}分  CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
