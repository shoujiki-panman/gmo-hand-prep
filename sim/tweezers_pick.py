"""新 Phase 2 試験7：G1（BrainCo の手つき）の右手のピンセットで豆をつまむ（物理あり）。箸（chopsticks_pick.py）と同じ条件で比べる用。

10/16 作り直し（合格ライン passline-tweezers-compare.md）：はさむ力・当たり・豆・追従 TRACK・移す TRANSFER・判定・戻り値は
chopsticks_pick.py の作りをそのまま写した（値も同じ）。ピンセット固有なのは次だけ:
- ピンセットは手に固定（先行研究と同じ前提）。本体を手のひらに溶接して引っぱる（mocap＋weld、0.2kg＋重力打ち消し）
- 2本の腕は後ろの端（親指と人差し指のくぼみ）のちょうつがいで回る。腕A＝親指側（upper）、腕B＝人差し指・中指側（lower）。+角度で閉じる
- 2本の腕の角度は等しくなるようにつなぐ（関節の等式。本物のピンセットはばねで左右対称に開くため。つながないと2本が同じ向きに流れる）
- 指（親指の曲げ・人差し指・中指）は腕の角度の狙いに合わせて曲げる（見た目だけ。はさむ力は一定トルクで出す）
- 腕・豆・皿・机は物理あり。G1 の体は見た目だけ（当たり判定なし、毎回角度を入れ直す）
- 腕は chopsticks_pick と同じ逆運動学で、ピンセットの先の真ん中を狙いへ動かし 10ms ごとに解く

  TRANSFER=1 BALL_CONDIM=6 T_HOLD=3 BALL_MU=0.6 BEAN_D_MM=9 .venv/bin/python tw_transfer.py base 5 1
"""

import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import chopsticks_look as cl
import chopsticks_pick as cp
import g1_demo_video as g1
import tweezers_look as tl

HERE = Path(__file__).parent
OUT = HERE / f"tweezers_pick_{os.environ.get('BEAN_D_MM', '9')}mm.mp4"
DEBUG = os.environ.get("DEBUG") == "1"
DEBUG_TIMES = (2.0, 2.6, 2.8, 3.0, 3.4, 4.0)
FONT = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", 28)

# ---- 箸と同じ値（chopsticks_pick.py と同じ環境変数・同じ既定・同じ式） ----
BEAN_R = float(os.environ.get("BEAN_D_MM", "9")) / 2000
BEAN_MASS = 0.0004 * (BEAN_R / 0.0045) ** 3
# ピンセットの場面の皿（10/16 本人OK で皿2を寄せた）。箸の CHOP_PLATES と同じやり方で、このファイルの中だけで決める（g1.PLATES は変えない）。
# 元の皿は (0.35,-0.10) のまま、皿2は +y に 20cm＝箸と同じ距離・同じ向き。g1.PLATES[1]＝(0.45,0.10) だと腕が届かなかった（ずれ 57〜81mm、10/16 実測）
TW_PLATES = [(0.35, -0.10), (0.30, 0.10)]   # 皿2 (0.35,0.10) は下げる段で腕のずれ 14〜58mm（10/16 実測）→ 届く範囲の実測で 2.5mm の (0.30,0.10)＝距離 20.6cm へ（本人OK）
PLATES = TW_PLATES
PLATE_XY = np.array(PLATES[0])
PLATE_TOP = g1.TABLE_TOP + 0.008
BEAN = np.array([*PLATE_XY, PLATE_TOP + BEAN_R])
GRIP_BACK = cp.GRIP_BACK                # はさむ点＝先端から 6mm 手前（逆運動学 cp.solve_arm もこの値を使う）
GRIP_KP = float(os.environ.get("GRIP_KP", "1.0"))
DOWN_DZ = float(os.environ.get("DOWN_DZ_MM", "0")) / 1000
PINCH_N = float(os.environ.get("PINCH_N", "0.3"))
HARD_SOLREF, HARD_SOLIMP = [0.004, 1], [0.95, 0.99, 0.001, 0.5, 2]
BALL_MU = float(os.environ.get("BALL_MU", "1.0"))
TRANSFER = os.environ.get("TRANSFER") == "1"
BALL_CONDIM = int(os.environ.get("BALL_CONDIM", "6"))   # 10/16 本人OKで既定を 6 に（4 だと皿の上で球が止まらない。実測）
T_HOLD = float(os.environ.get("T_HOLD", "0"))
NOSLIP = int(os.environ.get("NOSLIP", "2"))
TIP_LEN = 0.03                           # 先 3cm を硬い当たりの部分にする（太さは tweezers_look の腕のまま。細くしない＝④）
HINGE_DAMPING = 0.02                     # 箸のちょうつがいと同じ

# ---- ピンセット固有 ----
ARM_MAX = 0.06        # 腕1本の閉じる側の範囲（ラジアン）。+で閉じる
# 開く角度：斜めに下ろすと、先より手元側の腕が球の「頭」の高さを通る。その位置で
# 「球の半径＋腕の太さ＋3mm」の余裕が左右にできる角度にする（10/14）。
# 本物のピンセットは開きに限りがあるので、腕1本 0.08 ラジアン（先のすき間 約32mm）を上限にする
_reach = tl.LENGTH - GRIP_BACK - 2 * BEAN_R / np.sin(np.radians(tl.TILT_DEG))
_gap0 = 0.0126 * _reach / tl.LENGTH
OPEN_ANGLE = float(np.clip((2 * (BEAN_R + tl.ARM_R + 0.003) - _gap0) / (2 * _reach), 0.02, 0.08))
OPEN = -OPEN_ANGLE    # 腕の角度の狙い（開く）
CLOSE = float(os.environ.get("TW_CLOSE", "0.05"))   # 閉じきる狙い：先のすき間が約0.6mm になる角度（角度0で 12.6mm、腕1本1ラジアンで先が 120mm 動く）
FINGER_DELTA = {      # 指を「閉じる」側に曲げる量（ラジアン、見た目だけ）。g=+1 でこの分曲げ、g=−1 で同じ分伸ばす
    "right_thumb_proximal_joint": 0.15,
    "right_index_proximal_joint": 0.15,
    "right_middle_proximal_joint": 0.15,
}
ARM_JOINTS = cp.ARM_JOINTS


def finger_pose(theta):
    g = float(np.clip(-1 + 2 * (theta - OPEN) / (CLOSE - OPEN), -1, 1))
    return {n: v + (FINGER_DELTA.get(n, 0.0) * g) for n, v in tl.SHOW_POSE.items()}   # 見た目用の手の形（10/21。PINCH_POSE はピンセットの置き場所の計算用）


class Tw(tl.Tweezers):
    """tweezers_look.Tweezers に、開閉角での腕の位置（cp.solve_arm 用）と、はさむ点までの距離を足したもの。"""

    def __init__(self):
        super().__init__()
        self.signs = {"upper": 1.0, "lower": -1.0}
        rear, tip = self.lines["upper"]
        u = (tip - rear) / np.linalg.norm(tip - rear)
        self.grip_arm = float(np.linalg.norm(tip - GRIP_BACK * u - rear))   # 支点（後ろの端）→はさむ点（m）

    def local(self, theta):
        out = {}
        for k, (rear, tip) in self.lines.items():
            r = cp.rot(self.signs[k] * self.axis, theta)
            out[k] = [rear, rear + r @ (tip - rear)]
        return out

    def world(self, data, model, theta=0.0):
        b = model.body("right_base_link").id
        x, r = data.xpos[b], data.xmat[b].reshape(3, 3)
        return {k: [x + r @ p for p in v] for k, v in self.local(theta).items()}


def build(tw):
    spec = g1.build_spec(props=False, plates=PLATES)
    spec.option.timestep = 0.001
    spec.option.impratio = 10
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = NOSLIP
    for g in spec.geoms:                       # G1 は見た目だけ（世界に直接置いた机・皿・床は当たる）
        if g.parent.name != "world":
            g.contype = g.conaffinity = 0
        elif g.type == mujoco.mjtGeom.mjGEOM_CYLINDER:   # 皿を硬く。机・床は既定
            g.solref, g.solimp = HARD_SOLREF, HARD_SOLIMP
    w = spec.worldbody
    # ピンセットの本体＝手のひらと同じ座標の向き・位置に置き、手のひらに溶接する（箸の下の箸と同じ 0.2kg・重力打ち消し・溶接の硬さ）
    base = w.add_body(name="tw_base", gravcomp=1)
    base.add_freejoint()
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=tw.rear.tolist(), size=[0.004, 0, 0],
                  mass=0.2, contype=0, conaffinity=0, rgba=[0.6, 0.62, 0.66, 1])
    w.add_body(name="tw_target", mocap=True)
    spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="tw_base", name2="tw_target", solref=[0.005, 1],
                      data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    kw = dict(friction=[1.0, 0.01, 0.001], condim=4, rgba=[0.72, 0.74, 0.78, 1])
    for name in ("upper", "lower"):          # upper＝親指側、lower＝人差し指・中指側
        rear, tip = tw.lines[name]
        d = (tip - rear) / np.linalg.norm(tip - rear)
        mid = tip - TIP_LEN * d
        arm = base.add_body(name=f"tw_{name}", gravcomp=1)
        arm.add_joint(name=f"tw_{name}_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, pos=rear.tolist(),
                      axis=(tw.signs[name] * tw.axis).tolist(), damping=HINGE_DAMPING,
                      range=np.degrees([OPEN - 0.01, ARM_MAX]).tolist())   # この場面の角度の単位は度
        arm.add_geom(name=f"tw_{name}_thick", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*rear, *mid],
                     size=[tl.ARM_R, 0, 0], mass=0.005, **kw)
        # 先 3cm：箸の先と同じ硬い当たり＋condim 4・priority 1
        arm.add_geom(name=f"tw_{name}_tip", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*mid, *tip],
                     size=[tl.ARM_R, 0, 0], mass=0.005 * 0.002, solref=HARD_SOLREF, solimp=HARD_SOLIMP,
                     priority=1, **kw)
        pos = spec.add_actuator(name=f"tw_{name}_grip", target=f"tw_{name}_hinge", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        pos.set_to_position(kp=GRIP_KP)
        pos.ctrlrange = [OPEN - 0.01, ARM_MAX]
        pos.group = 0                          # 開くとき用（角度指定）
        frc = spec.add_actuator(name=f"tw_{name}_force", target=f"tw_{name}_hinge", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        frc.set_to_motor()
        frc.ctrlrange = [-1.0, 1.0]
        frc.group = 1                          # 閉じるとき用（一定トルク、+で閉じる）
    # 2本の腕を左右対称につなぐ（upper の角度 = lower の角度）
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="tw_upper_hinge", name2="tw_lower_hinge",
                      data=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # 2本の腕は後ろの付け根が同じ点で重なっているので、腕どうしだけは当てない（10/14 の途中版から。豆・皿・机との当たりは消していない）
    spec.add_exclude(bodyname1="tw_upper", bodyname2="tw_lower")
    beans = [BEAN] + [BEAN + [dx, dy, 0] for dx, dy in [(0.04, 0.03), (-0.035, 0.035), (0.03, -0.04)]]
    for i, p in enumerate(beans):
        bean = w.add_body(name=f"bean{i}", pos=p.tolist())
        bean.add_freejoint()
        bean.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BEAN_R, 0, 0], mass=BEAN_MASS,
                      friction=[BALL_MU, 0.01, 0.001], condim=BALL_CONDIM, rgba=[0.62, 0.42, 0.22, 1],
                      solref=HARD_SOLREF, solimp=HARD_SOLIMP, priority=2)
    model = spec.compile()
    return model, mujoco.MjData(model)


def main(bean=None, video=True, track=False):
    """bean: 豆の置き場所（x, y, z）。track: 2.0秒までは今の豆の位置から狙いを作り直す（chopsticks_pick と同じ）。"""
    global BEAN
    if bean is not None:
        BEAN = np.array(bean, float)
    tw = Tw()
    model, data = build(tw)

    def set_robot(q, theta):
        cp.set_arm(model, data, q)
        for n, v in finger_pose(theta).items():
            g1.setj(model, data, n, v)

    start = np.array([cl.ARM_HOLD[j] for j in ARM_JOINTS])
    set_robot(start, OPEN)
    mujoco.mj_kinematics(model, data)
    w0 = tw.world(data, model, OPEN)
    d0 = w0["lower"][1] - w0["lower"][0]
    horiz = np.array([d0[0], d0[1], 0]) / np.linalg.norm(d0[:2])
    ref_dir = np.cos(np.radians(tl.TILT_DEG)) * horiz + np.array([0, 0, -np.sin(np.radians(tl.TILT_DEG))])

    down_z = BEAN[2] + DOWN_DZ
    above_p = BEAN + [0, 0, 0.05]
    down_p = np.array([*BEAN[:2], down_z])
    lift_p = down_p + [0, 0, 0.05]
    plan = [(0.0, above_p, OPEN, "豆の上でピンセットを開く"),
            (1.0, above_p, OPEN, "豆の高さまで下げる"),
            (2.0, down_p, OPEN, "ピンセットを閉じて豆をはさむ"),
            (2.6, down_p, CLOSE, "はさんだまま持ち上げる"),
            (3.8, lift_p, CLOSE, "持ち上げたまま止める"),
            (4.8, lift_p, CLOSE, "")]
    rel = [1.0] * len(plan)
    if TRANSFER:
        p2 = np.array(PLATES[1], float)
        carry_p = np.array([*p2, lift_p[2]])
        low2_p = np.array([*p2, PLATE_TOP + BEAN_R + 0.01])
        up2_p = low2_p + [0, 0, 0.05]
        T_CARRY = float(os.environ.get("T_CARRY", "1.6"))
        t1 = 4.8 + T_CARRY
        plan = plan[:-1] + [(4.8, lift_p, CLOSE, "移す先へ運ぶ"),
                            (t1, carry_p, CLOSE, "下げる"),
                            (t1 + 0.6, low2_p, CLOSE, "ピンセットを開いて離す"),
                            (t1 + 1.1, low2_p, OPEN, "ピンセットを上げる"),
                            (t1 + 1.6, up2_p, OPEN, "豆が止まるのを待つ"),
                            (t1 + 1.6 + T_HOLD, up2_p, OPEN, "")]
        if T_HOLD <= 0: plan = plan[:-1]
        rel = [1.0] * 6 + [0.0] * (len(plan) - 6)
        release_t = t1 + 0.6

    # 最初の腕の角度：出発点を12通り変えて、豆の上でいちばんずれの小さい角度を選ぶ（ピンセット固有。1か所からだと解けなかった。10/14）
    rng = np.random.default_rng(0)
    best = None
    for k in range(12):
        s0 = start if k == 0 else start + rng.normal(0, 0.5, len(start))
        q, e, l = cp.solve_arm(model, data, tw, above_p, OPEN, s0, ref_dir, iters=300)
        if best is None or e + l < best[1] + best[2]:
            best = (q, e, l)
    q_now, c1, l1 = best
    print(f"最初の腕の角度（位置のずれ／先端の高さの差 mm）: {c1:.1f}/{l1:.1f}")
    print(f"開く角度 腕1本 {OPEN_ANGLE:.4f}rad → 先のすき間 {(tw.tip_gap + 2 * OPEN_ANGLE * tl.LENGTH) * 1000:.1f}mm（上限 約32mm）")
    if DEBUG:
        qd, e, l = cp.solve_arm(model, data, tw, down_p, OPEN, q_now, ref_dir, iters=300)
        set_robot(qd, OPEN)
        mujoco.mj_kinematics(model, data)
        for th in (OPEN, 0.0, CLOSE):
            w = tw.world(data, model, th)
            pt = {k: t - GRIP_BACK * (t - r) / np.linalg.norm(t - r) for k, (r, t) in w.items()}
            print(f"[形] 角度 {th:+.3f}  上のはさむ点-豆 {np.round((pt['upper'] - BEAN) * 1000, 1)}  下 {np.round((pt['lower'] - BEAN) * 1000, 1)}  (逆運動学 {e:.1f}mm/{l:.1f}mm)")
        print(f"[はさむ力] 支点→はさむ点 {tw.grip_arm * 1000:.1f}mm  PINCH_N {PINCH_N}N → トルク {PINCH_N * tw.grip_arm:.4f}N·m（腕1本ずつ）")
    state = {"q": q_now, "worst": 0.0}

    def at(t):
        for (t0, p0, a0, s0), (t1_, p1, a1, _), w0_, w1_ in zip(plan, plan[1:], rel, rel[1:]):
            if t <= t1_:
                k = np.clip((t - t0) / (t1_ - t0), 0, 1)
                k = k * k * (3 - 2 * k)
                return p0 + (p1 - p0) * k, a0 + (a1 - a0) * k, s0, w0_ + (w1_ - w0_) * k
        return plan[-1][1], plan[-1][2], plan[-2][3], rel[-1]

    seen = {"p": BEAN.copy()}
    TRACK_UNTIL = 2.0
    hinge_q = {n: model.joint(f"tw_{n}_hinge").qposadr[0] for n in ("upper", "lower")}

    def drive(t, solve=True):
        target, theta, label, w = at(t)
        if track:
            if solve and t < TRACK_UNTIL:
                seen["p"] = data.xpos[model.body("bean0").id].copy()
            target = target + (seen["p"] - BEAN) if w == 1.0 else target + (seen["p"] - BEAN) * w
        state["theta"] = theta
        if solve:
            # 腕は「開いた形のピンセット」を基準に解く（箸と同じ。左右対称なので先の真ん中は開閉で変わらない）
            state["q"], err, _ = cp.solve_arm(model, data, tw, target, OPEN, state["q"], ref_dir, iters=6)
            state["worst"] = max(state["worst"], err)
        data.qpos[robot_qpos] = rest[robot_qpos]
        set_robot(state["q"], theta)
        data.qvel[robot_dof] = 0
        mujoco.mj_kinematics(model, data)
        b = model.body("right_base_link").id
        m = model.body("tw_target").mocapid[0]
        data.mocap_pos[m] = data.xpos[b]
        data.mocap_quat[m] = data.xquat[b]
        # 閉じる段階（狙いが OPEN より閉じ側）は一定トルク、開く段階は角度指定。使わない側は group ごと止める（箸と同じ）
        closing = theta > OPEN + 1e-9 and not (TRANSFER and t >= release_t)
        model.opt.disableactuator = (1 << 0) if closing else (1 << 1)
        for n in ("upper", "lower"):
            data.ctrl[model.actuator(f"tw_{n}_grip").id] = 0.0 if closing else theta
            data.ctrl[model.actuator(f"tw_{n}_force").id] = PINCH_N * tw.grip_arm if closing else 0.0
        state["closing"] = closing
        return label

    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
              and not model.joint(j).name.startswith("tw_")]
    robot_qpos = np.array([model.jnt_qposadr[j] for j in hinges])
    robot_dof = np.array([model.jnt_dofadr[j] for j in hinges])
    rest = data.qpos.copy()

    drive(0.0)
    j = model.body("tw_base").jntadr[0]
    a = model.jnt_qposadr[j]
    m = model.body("tw_target").mocapid[0]
    data.qpos[a:a + 7] = [*data.mocap_pos[m], *data.mocap_quat[m]]
    for n in ("upper", "lower"):
        data.qpos[hinge_q[n]] = OPEN
    mujoco.mj_forward(model, data)

    W, H, FPS = 960, 720, 30
    hand_cam = mujoco.MjvCamera()
    hand_cam.lookat[:] = PLATE_XY.tolist() + [PLATE_TOP + 0.02]
    hand_cam.distance, hand_cam.azimuth, hand_cam.elevation = 0.28, 210, -20
    if TRANSFER:
        hand_cam.lookat[:] = ((PLATE_XY + np.array(PLATES[1])) / 2).tolist() + [PLATE_TOP + 0.02]
        hand_cam.distance = 0.55
    # 動画用カメラの向き（環境変数が無いときは上の値のまま）
    hand_cam.azimuth = float(os.environ.get("CAM_AZ", hand_cam.azimuth))
    hand_cam.elevation = float(os.environ.get("CAM_EL", hand_cam.elevation))
    hand_cam.distance = float(os.environ.get("CAM_DIST", hand_cam.distance))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", str(OUT)],
                          stdin=subprocess.PIPE) if video else None
    bean = model.body("bean0").id
    bean_j = model.body("bean0").jntadr[0]
    bean_da = model.jnt_dofadr[bean_j]
    steps = int(plan[-1][0] / model.opt.timestep)
    frame_every = int(1 / FPS / model.opt.timestep)

    held_from, max_h = None, 0.0
    tw_b = (model.body("tw_upper").id, model.body("tw_lower").id)

    def touching():
        return any(bean in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]) and
                   ({model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]} & set(tw_b))
                   for c in data.contact[:data.ncon])
    if TRANSFER:
        tr = {"lift_max": 0.0, "carry_min": 1.0, "lost_t": None, "lost_xyz": None, "t1": t1}
    r = mujoco.Renderer(model, H, W) if video else None
    for i in range(steps):
        t = i * model.opt.timestep
        label = drive(t, solve=(i % 10 == 0))
        mujoco.mj_step(model, data)
        h = data.xpos[bean][2] - (PLATE_TOP + BEAN_R)
        max_h = max(max_h, h)
        if t >= 3.8 and h > 0.02 and held_from is None:
            held_from = t
        if TRANSFER:
            if 3.8 <= t <= 4.8:
                tr["lift_max"] = max(tr["lift_max"], h)
            if 4.8 <= t <= t1:
                tr["carry_min"] = min(tr["carry_min"], h)
            if t >= 2.6 and i % 10 == 0:
                if touching():
                    tr["lost_t"] = None
                elif tr["lost_t"] is None:
                    tr["lost_t"], tr["lost_xyz"] = t, data.xpos[bean].copy()
        if DEBUG and 2.0 <= t <= (plan[-1][0] if TRANSFER else 4.8) and i % 100 == 0:
            st = []
            for ci, c in enumerate(data.contact[:data.ncon]):
                b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
                if bean in (b1, b2):
                    other = model.body(b2 if b1 == bean else b1).name or "world"
                    f = np.zeros(6); mujoco.mj_contactForce(model, data, ci, f)
                    st.append(f"{other}:{f[0]:.3f}N/{-c.dist * 1000:.2f}mm/μ{c.friction[0]:.2f}")
            ang = [round(float(data.qpos[hinge_q[n]]), 4) for n in ("upper", "lower")]
            print(f"[はさむ] t={t:.1f} 角度 {ang} "
                  f"{'トルク' if state['closing'] else '角度'} 豆の高さ {h * 1000:.1f}mm 接触(力/めり込み) {st}")
        if DEBUG and any(abs(t - x) < 5e-4 for x in DEBUG_TIMES):
            # ピンセットの先（実）と、手のひらから計算した先（狙い）のずれ・先と豆の位置
            goal = tw.world(data, model, state["theta"])
            tips = {}
            for n in ("upper", "lower"):
                gid = model.geom(f"tw_{n}_tip").id
                c, z = data.geom_xpos[gid], data.geom_xmat[gid].reshape(3, 3)[:, 2]
                e1, e2 = c + z * model.geom_size[gid][1], c - z * model.geom_size[gid][1]
                tips[n] = e1 if np.linalg.norm(e1 - goal[n][0]) > np.linalg.norm(e2 - goal[n][0]) else e2
            print(f"t={t:.2f} 腕の角度 {[round(float(data.qpos[hinge_q[n]]), 4) for n in ('upper', 'lower')]} 狙い {state['theta']:+.3f} "
                  f"先(実)-豆 mm 上{np.round((tips['upper'] - data.xpos[bean]) * 1000, 1)} 下{np.round((tips['lower'] - data.xpos[bean]) * 1000, 1)} "
                  f"先の高さ（皿の面から）mm {(tips['upper'][2] - PLATE_TOP) * 1000:.1f}/{(tips['lower'][2] - PLATE_TOP) * 1000:.1f}")
            parts = []
            for ci, c in enumerate(data.contact[:data.ncon]):
                b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
                if bean in (b1, b2):
                    other = model.body(b2 if b1 == bean else b1).name or "world"
                    f = np.zeros(6); mujoco.mj_contactForce(model, data, ci, f)
                    parts.append(f"{other}:{f[0]:.3f}N")
            print(f"t={t:.2f} 豆に触れている相手 {parts}  豆の位置 {np.round(data.xpos[bean] * 1000, 1)}")
            for n in ("upper", "lower"):
                ab = model.body(f"tw_{n}").id
                who = sorted({(model.geom(c.geom2 if model.geom_bodyid[c.geom1] == ab else c.geom1).name
                               or model.body(model.geom_bodyid[c.geom2] if model.geom_bodyid[c.geom1] == ab else model.geom_bodyid[c.geom1]).name or "world")
                              for c in data.contact[:data.ncon] if ab in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])})
                print(f"     腕 {n} が触れている相手 {who}")
        if video and i % frame_every == 0:
            r.update_scene(data, hand_cam)
            img = Image.fromarray(r.render())
            d = ImageDraw.Draw(img)
            d.rectangle([0, H - 96, W, H], fill=(30, 30, 35))
            d.text((20, H - 88), f"{t:4.1f}秒  {label}", font=FONT, fill=(255, 212, 0))
            d.text((20, H - 46), f"豆の高さ（皿から）{h * 1000:5.1f} mm", font=FONT, fill=(230, 230, 230))
            ff.stdin.write(np.asarray(img).tobytes())
    if video:
        ff.stdin.close()
        ff.wait()
    final_h = data.xpos[bean][2] - (PLATE_TOP + BEAN_R)
    ok = final_h > 0.02
    print(f"腕の狙いからのずれ（最大）: {state['worst']:.1f}mm")
    print(f"豆の高さ：最大 {max_h * 1000:.1f}mm・最後 {final_h * 1000:.1f}mm → {'成功' if ok else '失敗'}")
    if video:
        print(f"動画: {OUT}")
    if TRANSFER:
        p2 = np.array(PLATES[1], float)
        xy_err = float(np.linalg.norm(data.xpos[bean][:2] - p2))
        on_plate = xy_err <= 0.07 - BEAN_R                                         # (a)
        z_ok = abs(data.xpos[bean][2] - (PLATE_TOP + BEAN_R)) <= 0.003             # (b)
        free = not touching()                                                     # (c)
        lost = tr["lost_t"]
        print(f"移す: 皿2中心から {xy_err * 1000:.1f}mm 高さ {(data.xpos[bean][2] - PLATE_TOP - BEAN_R) * 1000:+.1f}mm ピンセットと接触 {not free} → {'移せた' if on_plate and z_ok and free else '移せず'}")
        return {"success": bool(tr["lift_max"] >= 0.02), "max_h": float(max_h), "final_h": float(final_h),
                "transferred": bool(on_plate and z_ok and free), "final_xy_err": xy_err * 1000,
                "lift_max": float(tr["lift_max"]), "carry_min": float(tr["carry_min"]), "worst": float(state["worst"]),
                "lost_t": lost, "lost_xyz": None if lost is None else tr["lost_xyz"].tolist(),
                "final_xyz": data.xpos[bean].tolist(),
                "final_speed": float(np.linalg.norm(data.qvel[bean_da:bean_da + 3])) * 1000, "flags": (bool(on_plate), bool(z_ok), bool(free)), "t1": t1}
    return {"success": bool(ok), "max_h": float(max_h), "final_h": float(final_h)}


if __name__ == "__main__":
    track = os.environ.get("TRACK") == "1"
    OUT = HERE / f"tweezers_pick_{os.environ.get('BEAN_D_MM', '9')}mm{'_track' if track else '_notrack'}.mp4"
    main(track=track)
