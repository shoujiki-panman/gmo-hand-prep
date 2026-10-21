"""試験10（10/16）：ばらつき入りのお手本で学習した ACT に、皿から皿2へ移させて測る。chop_eval.py を写して作った（元は変えない）。

- 評価もお手本と同じばらつき（摩擦 0.3〜1.0／直径 8〜10mm／質量 ×0.7〜1.3／位置 ±2cm）を、お手本（seed 0）とは別の seed 12345 で引く
- 観測＝腕7＋箸1＋画像2台（球の位置は入れない）。16手ずつ実行して見直す
- 成功＝transferred（皿2に置けた）。持ち上げ＝lift_max ≥ 2cm。失敗はどの段で落ちたかを書く（近づく/はさむ・運ぶ・離す）
- 1回目だけ動画 chop_eval_rand_<回数>[_perturb]_1.mp4

  cd ~/Developer/gmo-hand/lerobot && .venv/bin/python <このファイル> <pretrained_model のパス> [回数] [何手ずつ実行] [乱れ 秒,dx,dy（m）例 1.0,0.015,0]
"""

import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import chop_collect_rand as cr  # noqa: E402  （TRANSFER=1・T_HOLD=0 の設定と、ばらつきの引き方・読み直しを共用）

from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402


def stage(r, start_xy):
    """どの段で落ちたか（返り値からの判定・目安）"""
    if r["transferred"]:
        return "移せた"
    if r["lift_max"] < 0.02:
        moved = np.linalg.norm(np.array(r["final_xyz"][:2]) - start_xy) * 1000
        return f"近づく/はさむ（持ち上がらず・球が {moved:.0f}mm 動いた）"
    if r["carry_min"] < 0.01:
        return "運ぶ（途中で落とした）"
    return f"離す（皿2中心から {r['final_xy_err']:.0f}mm・{''.join('o' if f else 'x' for f in r['flags'])}）"


def main():
    path = Path(sys.argv[1]).expanduser()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_action_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    perturb = tuple(float(v) for v in sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    device = "mps"
    policy = ACTPolicy.from_pretrained(path).to(device).eval()
    policy.config.n_action_steps = n_action_steps
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(path),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    state_dim = policy.config.input_features["observation.state"].shape[0]
    assert state_dim == 8, "球の位置は観測に入れない（腕7＋箸1）"

    def act(ob):
        img = lambda a: torch.from_numpy(a).permute(2, 0, 1).float().div(255).unsqueeze(0)
        batch = {"observation.state": torch.from_numpy(ob["state"][:state_dim]).unsqueeze(0),
                 "observation.images.side": img(ob["images"]["side"]),
                 "observation.images.top": img(ob["images"]["top"])}
        with torch.no_grad():
            a = post(policy.select_action(pre(batch)))
        a = a.squeeze(0).cpu().numpy()
        return a[:7], float(a[7])

    rng = np.random.default_rng(12345)          # お手本（seed 0）とは別
    wins = lifts = 0
    t_all = time.time()
    tag = path.parent.name + ("_perturb" if perturb else "")
    for k in range(n):
        c = cr.sample(rng)
        got = cr.apply(c)
        cp = cr.cp
        policy.reset()
        pre.reset()
        post.reset()
        video = k < 1
        if video:
            cp.OUT = HERE / f"chop_eval_rand_{tag}_{k + 1}.mp4"
        bean = cp.BEAN + [*c["offset"], 0]
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            r = cp.main(bean=bean, policy=act, video=video, perturb=perturb)
        wins += r["transferred"]
        lifts += r["success"]
        print(f"第{k + 1}回: {'移せた' if r['transferred'] else '失敗'}  持ち上げ {r['lift_max'] * 1000:.0f}mm  "
              f"μ{got['model_mu']:.2f} 直径{got['cp_d_mm']:.1f}mm 質量{got['model_mass_g']:.2f}g ずらし {np.round(c['offset'] * 100, 1)}cm  "
              f"段: {stage(r, bean[:2] + (np.array(perturb[1:3]) if perturb else 0))}  （{time.time() - t0:.1f}秒）", flush=True)
    print(f"結果 {tag}: 移せた {wins}/{n}・持ち上げ {lifts}/{n}（{n_action_steps}手ずつ実行{f'、乱れ {perturb}' if perturb else ''}）"
          f"  所要 {(time.time() - t_all) / 60:.1f}分")


if __name__ == "__main__":
    main()
