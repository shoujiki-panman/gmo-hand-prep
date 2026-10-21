"""G1（BrainCo の手つき）の右手に箸を持たせた「見た目」を作る。Phase 3 の段1。

形だけ（関節の角度を直接入れる・物理なし）。箸は手の部品の子として付けるので、指を動かすと一緒に動く。
箸の寸法と持ち方は SIGGRAPH 2022「Learning to Use Chopsticks in Diverse Gripping Styles」に合わせた:
  長さ 26cm・半径 4mm のカプセル2本。標準の持ち方＝親指・人差し指・中指が上の箸、薬指が下の箸、
  下の箸は親指と人差し指の付け根のくぼみに当てる。
指先の位置を測ってから、その点を通るように箸を置く（2本の先端が一点で合わさるように向きを合わせる）。

  .venv/bin/python chopsticks_look.py   # chopsticks_look.png（全体と手元）ができる
"""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import g1_demo_video as g1

HERE = Path(__file__).parent
OUT = HERE / "chopsticks_look.png"
LENGTH, RADIUS = 0.26, 0.004
BEHIND = 0.03        # 箸の後ろの端が、持つ点からどれだけ後ろに出るか
TIP_GAP = 0.004      # 先端どうしのすき間（閉じかけ）

HOLD_POSE = {        # 箸を持つ手の形（ラジアン）
    "right_thumb_metacarpal_joint": 1.1,
    "right_thumb_proximal_joint": 0.35,
    "right_index_proximal_joint": 0.45,
    "right_middle_proximal_joint": 0.6,
    "right_ring_proximal_joint": 0.95,
    "right_pinky_proximal_joint": 1.3,
}

# 見た目用の手の形（10/21）。HOLD_POSE は箸を置く基準点（knuckle・valley）の計算に使うので変えない（変えると物理が変わる）。
# 映すときの指だけこちらにする（chopsticks_pick.set_arm）。3.5秒の場面で、親指の腹が下の箸に、人差し指・中指の腹が上の箸に
# 表面どうし約1mmで触れる角度を乱数＋山登りで探した（grip_fit.py）。上の箸が人差し指・中指と下の箸のあいだにあるので、2本は上の箸に当てた
SHOW_POSE = {
    "right_thumb_metacarpal_joint": 0.508,
    "right_thumb_proximal_joint": 0.926,
    "right_index_proximal_joint": 0.426,
    "right_middle_proximal_joint": 0.783,
    "right_ring_proximal_joint": 0.9,
    "right_pinky_proximal_joint": 1.2,
}


# 箸の先が豆の皿の上で斜め下を向く右腕の角度（関節を乱数で2万通り試して選んだ、10/14）。左腕は下ろす
ARM_HOLD = {
    "right_shoulder_pitch_joint": -0.642, "right_shoulder_roll_joint": -1.149,
    "right_shoulder_yaw_joint": 0.961, "right_elbow_joint": 0.508,
    "right_wrist_roll_joint": -1.831, "right_wrist_pitch_joint": -0.349, "right_wrist_yaw_joint": -1.157,
    "left_shoulder_pitch_joint": 0.2, "left_elbow_joint": 0.3,
}


def pose(model, data, arm=True):
    for name, v in {**g1.ARM_POSE, **HOLD_POSE, **(ARM_HOLD if arm else {})}.items():
        g1.setj(model, data, name, v)
    mujoco.mj_forward(model, data)


def chopstick_lines(model, data):
    """持つ形のときの指の位置から、2本の箸の両端（世界座標）を決める。"""
    p = lambda b: data.xpos[model.body(b).id].copy()
    knuckle = p("right_index_proximal_link")
    pinch = (p("right_index_tip") + p("right_middle_tip")) / 2
    valley = (p("right_thumb_metacarpal_link") + knuckle) / 2
    ring = p("right_ring_tip")

    up_dir = (pinch - knuckle) / np.linalg.norm(pinch - knuckle)
    up_rear = knuckle - BEHIND * up_dir
    up_tip = up_rear + LENGTH * up_dir

    side = np.cross(up_dir, ring - knuckle)
    side /= np.linalg.norm(side)
    lo_rear = valley - BEHIND * (ring - valley) / np.linalg.norm(ring - valley)
    lo_dir = (up_tip + TIP_GAP * side - lo_rear)
    lo_dir /= np.linalg.norm(lo_dir)
    lo_tip = lo_rear + LENGTH * lo_dir
    return {"upper": (up_rear, up_tip, "right_index_proximal_link"),
            "lower": (lo_rear, lo_tip, "right_base_link")}


def build():
    # 1回目：持つ形にして指の位置を測る
    model, data = g1.build()
    pose(model, data, arm=False)   # 箸の付け方は手の形だけで決まる（腕の角度に関係しない）
    lines = chopstick_lines(model, data)
    frames = {b: (data.xpos[model.body(b).id].copy(), data.xmat[model.body(b).id].reshape(3, 3).copy())
              for _, _, b in lines.values()}

    # 2回目：測った位置に箸を足して作り直す（親の部品の座標に直して付ける）
    spec = g1.build_spec()
    for name, (rear, tip, parent) in lines.items():
        x, r = frames[parent]
        local = lambda q: r.T @ (q - x)
        spec.body(parent).add_geom(
            name=f"chopstick_{name}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[*local(rear), *local(tip)], size=[RADIUS, 0, 0],
            contype=0, conaffinity=0, rgba=[0.80, 0.62, 0.38, 1])
    model = spec.compile()
    data = mujoco.MjData(model)
    pose(model, data)
    return model, data


def main():
    model, data = build()
    tip = lambda n: data.geom_xpos[model.geom(n).id]
    up, lo = model.geom("chopstick_upper").id, model.geom("chopstick_lower").id
    ends = []
    for gid in (up, lo):
        c, z = data.geom_xpos[gid], data.geom_xmat[gid].reshape(3, 3)[:, 2]
        ends.append(c + z * model.geom_size[gid][1])
        ends.append(c - z * model.geom_size[gid][1])
    print("箸の先端どうしの距離（近い組）: %.1f mm" % (1000 * min(
        np.linalg.norm(a - b) for a in ends[:2] for b in ends[2:])))

    W, H = model.vis.global_.offwidth, model.vis.global_.offheight
    hand = data.xpos[model.body("right_base_link").id]
    shots = []
    with mujoco.Renderer(model, H, W) as r:
        for look, dist, az, el in [([0.3, -0.1, 0.85], 1.6, 150, -20),
                                   (hand + [0.05, 0, -0.06], 0.45, 220, -20),
                                   (hand + [0.05, 0, -0.06], 0.45, 300, -15)]:
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
