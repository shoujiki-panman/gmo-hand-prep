"""Phase 2：G1（BrainCo の手つき）の右手にピンセットを持たせた「見た目」を作る（形だけ・物理なし）。

先行研究（Xu ほか, IROS 2025, arXiv 2507.06822）に合わせた持ち方:
  ピンセットは最初から手に固定。人差し指が片方の腕に掛かり、親指・人差し指・中指で開き具合を調整する。
ここでは、腕の後ろの端を親指と人差し指の付け根のくぼみで合わせ、
  腕A＝くぼみ → 親指の指先、腕B＝くぼみ → 人差し指と中指の指先の間、を通す。長さ 12cm。
腕の角度は chopsticks_pick と同じ逆運動学で、ピンセットの先が豆の皿の上で約50度下を向くように解く。

  .venv/bin/python tweezers_look.py   # tweezers_look.png（全体と手元2方向）ができる
"""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import chopsticks_look as cl
import chopsticks_pick as cp
import g1_demo_video as g1

HERE = Path(__file__).parent
OUT = HERE / "tweezers_look.png"
LENGTH = 0.12          # ピンセットの長さ
ARM_R = 0.0022         # 腕の太さ（見た目用の丸棒。実物は幅 8mm・厚さ 1.5mm ほどの板）
TILT_DEG = 35          # ピンセットの下向きの角度（50度だと手が届かなかった。10/14）

PINCH_POSE = {         # つまむ手の形（ラジアン）。親指の先と人差し指・中指の先が 12mm になる形を乱数で探した（10/14）
    "right_thumb_metacarpal_joint": 1.35,
    "right_thumb_proximal_joint": 0.819,
    "right_index_proximal_joint": 0.807,
    "right_middle_proximal_joint": 0.391,
    "right_ring_proximal_joint": 1.3,
    "right_pinky_proximal_joint": 1.3,
}

# 見た目用の手の形（10/21）。PINCH_POSE はピンセットを置く基準点（親指の先・人差し指の先）の計算に使うので変えない（変えると物理が変わる）。
# 映すときの指だけこちらにする（tweezers_pick.finger_pose）。閉じた形（＋FINGER_DELTA）で、親指の腹が親指側の腕に、
# 人差し指・中指の腹が反対の腕に表面どうし約1mmで触れる角度を、3.5秒の場面で乱数＋山登りで探した（grip_fit.py）。薬指・小指は軽く曲げて添える
SHOW_POSE = {
    "right_thumb_metacarpal_joint": 1.345,
    "right_thumb_proximal_joint": 0.616,   # 閉じた形で 0.766
    "right_index_proximal_joint": 0.411,   # 閉じた形で 0.561
    "right_middle_proximal_joint": 0.372,  # 閉じた形で 0.522
    "right_ring_proximal_joint": 1.0,
    "right_pinky_proximal_joint": 1.1,
}


class Tweezers:
    """手のひら（right_base_link）から見たピンセットの2本の腕（chopsticks_pick.solve_arm で使える形）。"""

    def __init__(self):
        model, data = g1.build()
        for name, v in {**g1.ARM_POSE, **PINCH_POSE}.items():
            g1.setj(model, data, name, v)
        mujoco.mj_forward(model, data)
        b = model.body("right_base_link").id
        x, r = data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy()
        p = lambda n: r.T @ (data.xpos[model.body(n).id] - x)
        rear = (p("right_thumb_metacarpal_link") + p("right_index_proximal_link")) / 2
        thumb = p("right_thumb_tip")
        finger = (p("right_index_tip") + p("right_middle_tip")) / 2
        da = (thumb - rear) / np.linalg.norm(thumb - rear)
        db = (finger - rear) / np.linalg.norm(finger - rear)
        self.rear = rear
        self.lines = {"upper": (rear, rear + LENGTH * da), "lower": (rear, rear + LENGTH * db)}
        self.axis = np.cross(da, db) / np.linalg.norm(np.cross(da, db))
        self.tip_gap = np.linalg.norm(self.lines["upper"][1] - self.lines["lower"][1])

    def world(self, data, model, theta=0.0):
        b = model.body("right_base_link").id
        x, r = data.xpos[b], data.xmat[b].reshape(3, 3)
        return {k: [x + r @ q for q in v] for k, v in self.lines.items()}


def build(tw):
    spec = g1.build_spec()
    base = spec.body("right_base_link")
    for name, (rear, tip) in tw.lines.items():
        base.add_geom(name=f"tweezers_{name}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[*rear, *tip], size=[ARM_R, 0, 0], contype=0, conaffinity=0,
                      rgba=[0.72, 0.74, 0.78, 1])
    base.add_geom(name="tweezers_joint", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=tw.rear.tolist(),
                  size=[0.004, 0, 0], contype=0, conaffinity=0, rgba=[0.6, 0.62, 0.66, 1])
    model = spec.compile()
    return model, mujoco.MjData(model)


def pose(model, data, q):
    cp.set_arm(model, data, q)
    for name, v in PINCH_POSE.items():
        g1.setj(model, data, name, v)


def main():
    tw = Tweezers()
    print(f"ピンセットの先のすき間（つまむ形のとき）: {tw.tip_gap * 1000:.1f} mm")
    model, data = build(tw)

    start = np.array([cl.ARM_HOLD[j] for j in cp.ARM_JOINTS])
    pose(model, data, start)
    mujoco.mj_kinematics(model, data)
    w0 = tw.world(data, model)
    d0 = (w0["lower"][1] - w0["lower"][0])
    horiz = np.array([d0[0], d0[1], 0]) / np.linalg.norm(d0[:2])
    ref_dir = np.cos(np.radians(TILT_DEG)) * horiz + np.array([0, 0, -np.sin(np.radians(TILT_DEG))])
    target = cp.BEAN + [0, 0, 0.02]
    # chopsticks_pick.set_arm は箸の手の形も入れるので、解いたあとでつまむ手の形に入れ直す
    # 出発点を何通りか変えて、いちばんずれの小さい腕の角度を選ぶ（1か所からだと解けないことがあった）
    rng = np.random.default_rng(0)
    best = None
    for k in range(12):
        s0 = start if k == 0 else start + rng.normal(0, 0.5, len(start))
        q, err, level = cp.solve_arm(model, data, tw, target, 0.0, s0, ref_dir, iters=300)
        if best is None or err + level < best[1] + best[2]:
            best = (q, err, level)
    q, err, level = best
    print(f"腕の角度（位置のずれ／先の高さの差 mm）: {err:.1f}/{level:.1f}")
    pose(model, data, q)
    mujoco.mj_forward(model, data)

    W, H = model.vis.global_.offwidth, model.vis.global_.offheight
    hand = data.xpos[model.body("right_base_link").id]
    shots = []
    with mujoco.Renderer(model, H, W) as r:
        for look, dist, az, el in [([0.3, -0.1, 0.85], 1.6, 150, -20),
                                   (hand + [0.04, 0, -0.06], 0.38, 220, -20),
                                   (hand + [0.04, 0, -0.06], 0.38, 300, -10)]:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = look
            cam.distance, cam.azimuth, cam.elevation = dist, az, el
            r.update_scene(data, cam)
            shots.append(Image.fromarray(r.render()).resize((W // 2, H // 2)))
    sheet = Image.new("RGB", (W // 2 * 3, H // 2))
    for i, s in enumerate(shots):
        sheet.paste(s, (i * W // 2, 0))
    sheet.save(OUT)
    print(f"保存: {OUT}")


if __name__ == "__main__":
    main()
