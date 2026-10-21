"""Phase 4b 段1「握って保つ」（10/21・学習なし・物理あり）。合格ライン passline-grip-hold.md。

道具を手に溶接せず、指の接触だけで保持できるかを測る。
- 持ち方は grip_design.json の値そのまま: ピンセット＝tw 案1（ペン持ち #20）、箸＝chop 案3（標準 #21）
- 指: 当たり判定は「手の部品 × 道具」の <pair> だけ（geom 側は contype/conaffinity 0。指どうし・指と手のひら・腕・体・机は当たらない）
- 指の関節: 主関節6つに position actuator（kp＝KP、forcerange＝Unitree モデルの actuatorfrcrange）。
  遠位関節は MIMIC（URDF の倍率）を joint equality で連動（このモデルは compile しても equality が0本＝自動では入らない。実測）
- 道具: free joint。ピンセット＝腕A（本体）＋腕B（後ろの端のちょうつがい・ばね）。箸＝2本別々の free body（箸どうしは当てない）
- 腕・体: 指令した角度をそのまま入れる（毎ステップ qpos と qvel を上書き）。指の反力で腕が動かないよう、腕・体の関節は armature を大きく・gravcomp
- 初期配置は keyframe（指の角度・道具の位置を先に決める）。道具に ±1mm・±1度の乱れ（seed 2026）
- 手順: 0〜5秒 止める → 5〜7秒 手のひらを 上・左・下・右 に 10cm ずつ（各0.5秒、元の位置へ戻る）→ 7〜12秒 止める
- 0.1秒ごとに、道具の重心の手のひら座標でのずれ（mm）と回転（度）、接触の数と力（指ごと）を CSV に記録

  cd ~/Developer/gmo-hand/lerobot && TOOL=tweezers .venv/bin/python <このフォルダ>/grip_hold.py base 5 1
  cd ~/Developer/gmo-hand/lerobot && TOOL=chopsticks .venv/bin/python <このフォルダ>/grip_hold.py base 5 1
  引数: 名前 回数 動画(1/0。1回目だけ撮る)

環境変数（往復2回目の調整用。既定は世の中の定番の値）:
  KP=5 PRESS=0.05 MU=1 IMPRATIO=10 NOSLIP=0 SOLIMP=0.99,0.99,0.01 SOLREF=0.01,1 DT=0.001 HINGE_K=0.3 MOVE_CM=10
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
import chopsticks_look as cl  # noqa: E402
import g1_demo_video as g1  # noqa: E402

TOOL = os.environ.get("TOOL", "tweezers")
assert TOOL in ("tweezers", "chopsticks"), TOOL
KEY, VARIANT = ("tw", 1) if TOOL == "tweezers" else ("chop", 3)   # 本人決定 10/21：ピンセット案1・箸案3
DESIGN = next(v for v in json.loads((HERE / "grip_design.json").read_text())[KEY] if v["variant"] == VARIANT)

KP = float(os.environ.get("KP", "5"))
PRESS = float(os.environ.get("PRESS", "0.05"))
MU = float(os.environ.get("MU", "1"))
IMPRATIO = float(os.environ.get("IMPRATIO", "10"))
NOSLIP = int(os.environ.get("NOSLIP", "0"))
SOLIMP = [float(x) for x in os.environ.get("SOLIMP", "0.99,0.99,0.01").split(",")] + [0.5, 2]
SOLREF = [float(x) for x in os.environ.get("SOLREF", "0.01,1").split(",")]
DT = float(os.environ.get("DT", "0.001"))
HINGE_K = float(os.environ.get("HINGE_K", "0.3"))   # ピンセットのばね N·m/rad（実物の値は未確認・推測）
MOVE = float(os.environ.get("MOVE_CM", "10")) / 100

T_HOLD1, T_MOVE, T_HOLD2 = 5.0, 2.0, 5.0
T_END = T_HOLD1 + T_MOVE + T_HOLD2
LIMIT_MM, LIMIT_DEG = 5.0, 5.0

# 道具の寸法（grip_design.py と同じ）
TW_LEN, TW_R, TW_ARM_MASS = 0.12, 0.0022, 0.005          # 腕1本 5g（tweezers_pick と同じ）
CH_LEN, CH_R, CH_TIP_R, CH_MASS = 0.26, 0.004, 0.00125, 0.006   # 1本 6g（竹 26cm の目安・推測）

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MAIN = ["right_thumb_metacarpal_joint", "right_thumb_proximal_joint", "right_index_proximal_joint",
        "right_middle_proximal_joint", "right_ring_proximal_joint", "right_pinky_proximal_joint"]
PRESS_J = (["right_thumb_proximal_joint", "right_index_proximal_joint", "right_middle_proximal_joint"]
           + (["right_ring_proximal_joint"] if KEY == "chop" else []))   # 道具に当てる指だけ押しつける
TIP_JOINTS = ["right_thumb_tip"] + [f"right_{f}_tip_joint" for f in FINGERS[1:]]
ARM_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
              "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
FONT = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", 26)


def is_hand_body(name):
    return name.startswith("right_") and (name in ("right_base_link", "right_base2_link")
                                          or any(f in name for f in FINGERS))


def finger_of(name):
    for f in FINGERS:
        if f in name:
            return f
    return "palm"


def seg_dist(p1, q1, p2, q2):
    """線分どうしの最短距離（サンプリング・十分細かい）。"""
    s = np.linspace(0, 1, 400)[:, None]
    a = p1 + s * (q1 - p1)
    b = p2 + s * (q2 - p2)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(d.min())


def chop_tip_len(lines):
    """太い部分どうしの重なり（案3 は −1.77mm）を、細い部分を長くして無くす（合格ライン⑥）。5mm 刻みで 0.5mm 以上離れる最短の長さ。"""
    lo_r, lo_t = (np.array(lines["lower"][k]) for k in ("rear", "tip"))
    up_r, up_t = (np.array(lines["upper"][k]) for k in ("rear", "tip"))
    for L in np.arange(0.030, 0.121, 0.005):
        lo_m = lo_t - L * (lo_t - lo_r) / np.linalg.norm(lo_t - lo_r)
        up_m = up_t - L * (up_t - up_r) / np.linalg.norm(up_t - up_r)
        gap = seg_dist(lo_r, lo_m, up_r, up_m) - 2 * CH_R
        if gap >= 0.0005:
            return float(L), gap
    raise RuntimeError("細い部分を 12cm にしても太い部分が重なる")


# ---------------------------------------------------------------- 場面
def build():
    spec = g1.build_spec(props=False)
    o = spec.option
    o.timestep, o.impratio, o.noslip_iterations = DT, IMPRATIO, NOSLIP
    o.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for g in spec.geoms:                      # 全部いったん当たり無し。当たりは下の <pair> だけ
        g.contype = g.conaffinity = 0
    hand_geoms = []
    for b in spec.bodies:
        if b.name and is_hand_body(b.name):
            for g in b.geoms:
                if g.group == 0 or g.group == 3:   # URDF の collision 側
                    hand_geoms.append(g)
        elif b.name and b.name != "world":
            b.gravcomp = 1                   # 腕・体は指令どおりに動かす（重力で落ちない）
    for j in spec.joints:
        if j.name.startswith("right_") and any(f in j.name for f in FINGERS):
            j.damping, j.armature = [0.05, 0, 0], 0.001    # 箸論文の手の関節と同じ
        else:
            j.armature = 1000.0               # 腕・体は指の反力で動かない（角度は毎ステップ上書き）
    # MIMIC：遠位＝倍率×近位。指先の固定関節は 0 に固定
    for prox, (dist, mult) in g1.MIMIC.items():
        spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=dist, name2=prox, solref=[0.005, 1],
                          data=[0, mult, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for n in TIP_JOINTS:
        spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=n, solref=[0.005, 1],
                          data=[0] * 11)
    # 指の position actuator（主関節だけ）
    for n in MAIN:
        a = spec.add_actuator(name=f"act_{n}", target=n, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=KP)
        fr = 0.5 if "metacarpal" in n else 1.1 if "thumb" in n else 2.0   # Unitree モデルの actuatorfrcrange
        a.forcelimited = 1
        a.forcerange = [-fr, fr]
        a.ctrlrange = [0, 1.6]

    w = spec.worldbody
    tool_geoms, tool_bodies, info = [], [], {}
    L = DESIGN["lines"]
    if KEY == "tw":
        rear = np.array(L["A"]["rear"]); ta = np.array(L["A"]["tip"]); tb = np.array(L["B"]["tip"])
        body = w.add_body(name="tool_a")
        body.add_freejoint(name="tool_a_free")
        tool_geoms.append(body.add_geom(name="tw_A", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*rear, *ta],
                                        size=[TW_R, 0, 0], mass=TW_ARM_MASS, rgba=[0.72, 0.74, 0.78, 1]))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=rear.tolist(), size=[0.004, 0, 0], mass=0.001,
                      rgba=[0.6, 0.62, 0.66, 1])
        bb = body.add_body(name="tool_b")
        da, db = (ta - rear) / TW_LEN, (tb - rear) / TW_LEN
        ax = np.cross(db, da); ax /= np.linalg.norm(ax)          # +角度で B が A へ近づく（閉じる）
        open_ang = float(np.arccos(np.clip(da.dot(db), -1, 1)))
        bb.add_joint(name="tw_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, pos=rear.tolist(), axis=ax.tolist(),
                     stiffness=[HINGE_K, 0, 0], damping=[0.01, 0, 0], limited=1,
                     range=np.degrees([-0.05, open_ang - 2 * TW_R / TW_LEN * 0.9]).tolist())
        tool_geoms.append(bb.add_geom(name="tw_B", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*rear, *tb],
                                      size=[TW_R, 0, 0], mass=TW_ARM_MASS, rgba=[0.72, 0.74, 0.78, 1]))
        tool_bodies = ["tool_a"]          # 測るのは本体（腕A）。腕B は本体に対するちょうつがいの角度で別に見る
        free_bodies = ["tool_a"]
        info["axis_palm"] = np.array(DESIGN["axis_palm"])
        info["open_ang"] = open_ang
    else:
        tip_len, gap = chop_tip_len(L)
        info["tip_len"], info["thick_gap"] = tip_len, gap
        free_bodies = []
        for k, nm in (("lower", "chop_lo"), ("upper", "chop_up")):
            r, t = np.array(L[k]["rear"]), np.array(L[k]["tip"])
            d = (t - r) / np.linalg.norm(t - r)
            m = t - tip_len * d
            body = w.add_body(name=nm)
            body.add_freejoint(name=f"{nm}_free")
            tool_geoms.append(body.add_geom(name=f"{nm}_thick", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*r, *m],
                                            size=[CH_R, 0, 0], mass=CH_MASS, rgba=[0.80, 0.62, 0.38, 1]))
            tool_geoms.append(body.add_geom(name=f"{nm}_tip", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*m, *t],
                                            size=[CH_TIP_R, 0, 0], mass=CH_MASS * 0.05, rgba=[0.80, 0.62, 0.38, 1]))
            free_bodies.append(nm)
        tool_bodies = free_bodies
        info["axis_palm"] = np.array(DESIGN["axis_palm"])
    # 手の部品 × 道具 の組だけ当てる（箸論文と同じ condim 4・friction 1 0.005 0.0001・solimp 0.99 0.99 0.01・solref 0.01 1）
    for hg in hand_geoms:
        for tg in tool_geoms:
            spec.add_pair(geomname1=hg.name or "", geomname2=tg.name, condim=4,
                          friction=[MU, MU, 0.005, 0.0001, 0.0001], solref=SOLREF, solimp=SOLIMP, margin=0, gap=0)
    return spec, tool_bodies, free_bodies, info


def name_hand_geoms(spec):
    """<pair> は geom の名前で組むので、名前の無い手の当たり判定に名前を付ける。"""
    k = 0
    for b in spec.bodies:
        if b.name and is_hand_body(b.name):
            for g in b.geoms:
                if g.group in (0, 3) and not g.name:
                    g.name = f"hand_col_{b.name}_{k}"
                    k += 1


def compile_with_key():
    # 名前付け → pair を組み直すため、もう一度作る
    spec, tool_bodies, free_bodies, info = None, None, None, None
    orig_build_spec = g1.build_spec

    def named_build_spec(*a, **kw):
        s = orig_build_spec(*a, **kw)
        name_hand_geoms(s)
        return s
    g1.build_spec = named_build_spec
    try:
        spec, tool_bodies, free_bodies, info = build()
    finally:
        g1.build_spec = orig_build_spec
    m = spec.compile()
    d = mujoco.MjData(m)
    # keyframe：腕・指の角度と道具の位置を先に決める
    for n, v in {**g1.ARM_POSE, **cl.ARM_HOLD}.items():
        g1.setj(m, d, n, v)
    for n, v in DESIGN["hand_pose"].items():
        g1.setj(m, d, n, v)
    mujoco.mj_kinematics(m, d)
    b = m.body("right_base_link").id
    for fb in free_bodies:
        a = m.jnt_qposadr[m.body(fb).jntadr[0]]
        d.qpos[a:a + 7] = [*d.xpos[b], *d.xquat[b]]          # 道具の座標＝手のひら座標（geom は手のひら座標で作った）
    ctrl = np.array([DESIGN["hand_pose"][n] + (PRESS if n in PRESS_J else 0.0) for n in MAIN])
    spec.add_key(name="grip", qpos=d.qpos.tolist(), ctrl=ctrl.tolist())
    m = spec.compile()
    return m, tool_bodies, free_bodies, info


# ---------------------------------------------------------------- 腕（手のひらを 10cm 動かす）
def offset(t):
    """手のひらの狙いのずれ（世界座標 m）。5〜7秒で 上→左→下→右（各0.5秒・なめらか）。"""
    if t <= T_HOLD1 or t >= T_HOLD1 + T_MOVE:
        return np.zeros(3)
    pts = [np.zeros(3), np.array([0, 0, MOVE]), np.array([0, MOVE, MOVE]), np.array([0, MOVE, 0]), np.zeros(3)]
    s = (t - T_HOLD1) / (T_MOVE / 4)
    k = min(int(s), 3)
    u = s - k
    u = u * u * (3 - 2 * u)
    return pts[k] + (pts[k + 1] - pts[k]) * u


def arm_traj(m):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_kinematics(m, d)
    b = m.body("right_base_link").id
    p0, R0 = d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()
    qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in ARM_JOINTS])
    dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in ARM_JOINTS])
    lo = np.array([m.jnt_range[m.joint(n).id][0] for n in ARM_JOINTS])
    hi = np.array([m.jnt_range[m.joint(n).id][1] for n in ARM_JOINTS])
    ts = np.arange(0, T_END + 0.0101, 0.01)
    q = d.qpos[qadr].copy()
    out, worst = [], 0.0
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for t in ts:
        target = p0 + offset(t)
        for _ in range(30):
            d.qpos[qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.xmat[b].reshape(3, 3)
            ep = target - d.xpos[b]
            er = 0.5 * sum(np.cross(R[:, i], R0[:, i]) for i in range(3))
            e = np.concatenate([ep, er])
            if np.linalg.norm(ep) < 1e-5 and np.linalg.norm(er) < 1e-4:
                break
            mujoco.mj_jacBody(m, d, jp, jr, b)
            J = np.vstack([jp[:, dadr], jr[:, dadr]])
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), lo, hi)
        d.qpos[qadr] = q
        mujoco.mj_kinematics(m, d)
        worst = max(worst, float(np.linalg.norm(target - d.xpos[b])))
        out.append(q.copy())
    return ts, np.array(out), qadr, dadr, worst


# ---------------------------------------------------------------- 測る
def rel_pose(m, d, body):
    b = m.body("right_base_link").id
    x, R = d.xpos[b], d.xmat[b].reshape(3, 3)
    tb = m.body(body).id
    com = R.T @ (d.xipos[tb] - x)
    Rt = R.T @ d.xmat[tb].reshape(3, 3)
    return com, Rt


def rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def contacts(m, d, tool_ids):
    per = {f: 0.0 for f in (*FINGERS, "palm")}
    n = 0
    f6 = np.zeros(6)
    for i, c in enumerate(d.contact[:d.ncon]):
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 in tool_ids or b2 in tool_ids:
            other = b2 if b1 in tool_ids else b1
            mujoco.mj_contactForce(m, d, i, f6)
            per[finger_of(m.body(other).name)] += f6[0]
            n += 1
    return n, per


def perturb(m, d, free_bodies, rng):
    for fb in free_bodies:
        a = m.jnt_qposadr[m.body(fb).jntadr[0]]
        tb = m.body(fb).id
        # 重心のまわりに ±1度（ランダムな軸）、位置 ±1mm（各軸）
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        ang = np.radians(rng.uniform(-1, 1))
        dq = np.zeros(4); mujoco.mju_axisAngle2Quat(dq, axis, ang)
        mujoco.mj_kinematics(m, d)
        com = d.xipos[tb].copy()
        pos = d.qpos[a:a + 3].copy()
        R = np.zeros(9); mujoco.mju_quat2Mat(R, dq); R = R.reshape(3, 3)
        newq = np.zeros(4); mujoco.mju_mulQuat(newq, dq, d.qpos[a + 3:a + 7].copy())
        d.qpos[a:a + 3] = com + R @ (pos - com) + rng.uniform(-0.001, 0.001, 3)
        d.qpos[a + 3:a + 7] = newq
    mujoco.mj_forward(m, d)


def min_gap_mm(m, d, tool_ids):
    """手の当たり判定と道具の表面の最短距離（mm、負＝めり込み）。"""
    ft = np.zeros(6)
    best = 1e9
    for p in range(m.npair):
        g1_, g2_ = m.pair_geom1[p], m.pair_geom2[p]
        best = min(best, mujoco.mj_geomDistance(m, d, g1_, g2_, 0.05, ft) * 1000)
    return best


def run(m, traj, tool_bodies, free_bodies, info, rng, video_path=None, csv_rows=None, run_id=0):
    ts, Q, qadr, dadr, _ = traj
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    perturb(m, d, free_bodies, rng)
    gap0 = min_gap_mm(m, d, None)
    d.ctrl[:] = m.key_ctrl[0]
    hand_dof = set()
    for j in range(m.njnt):
        n = m.joint(j).name
        if (n.startswith("right_") and any(f in n for f in FINGERS)) or n.endswith("_free") or n == "tw_hinge":
            hand_dof.add(j)
    kin = [j for j in range(m.njnt) if j not in hand_dof]
    kin_q = np.array([m.jnt_qposadr[j] for j in kin])
    kin_v = np.array([m.jnt_dofadr[j] for j in kin])
    rest = d.qpos.copy()
    tool_ids = {m.body(b).id for b in (free_bodies + (["tool_b"] if KEY == "tw" else []))}
    ref = {b: rel_pose(m, d, b) for b in tool_bodies}
    hinge_adr = m.jnt_qposadr[m.joint("tw_hinge").id] if KEY == "tw" else None

    W, H, FPS = 960, 720, 30
    ff = r = None
    if video_path:
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", str(video_path)],
                              stdin=subprocess.PIPE)
        r = mujoco.Renderer(m, H, W)
        cam = mujoco.MjvCamera()
        mujoco.mj_kinematics(m, d)
        bb = m.body("right_base_link").id
        a = d.xmat[bb].reshape(3, 3) @ info["axis_palm"]
        h = np.cross(a, [0, 0, 1.0]); h /= np.linalg.norm(h)       # 道具の軸に直交する水平の向き＝横から
        cam.azimuth = float(np.degrees(np.arctan2(h[1], h[0])))
        cam.elevation, cam.distance = -15, 0.22 if KEY == "tw" else 0.36
        look_local = np.array(next(iter(DESIGN["contacts"].values()))["tool_pt"])
        look_local = (look_local + np.array(DESIGN["contacts"][list(DESIGN["contacts"])[1]]["tool_pt"])) / 2
    steps = int(round(T_END / DT))
    rec_every = int(round(0.1 / DT))
    frame_every = int(round(1 / FPS / DT))
    worst_mm, worst_deg, lost = 0.0, 0.0, None
    last = {}
    for i in range(steps + 1):
        t = i * DT
        k = min(int(t / 0.01), len(ts) - 2)
        u = (t - ts[k]) / 0.01
        q = Q[k] + (Q[k + 1] - Q[k]) * u
        dq = (Q[k + 1] - Q[k]) / 0.01
        d.qpos[kin_q] = rest[kin_q]
        d.qvel[kin_v] = 0
        d.qpos[qadr] = q
        d.qvel[dadr] = dq
        if i % rec_every == 0 or (video_path and i % frame_every == 0):
            mujoco.mj_forward(m, d)
            vals = {}
            for b in tool_bodies:
                com, Rt = rel_pose(m, d, b)
                dp = (com - ref[b][0]) * 1000
                vals[b] = (dp, rot_deg(ref[b][1], Rt))
            mm = max(np.linalg.norm(v[0]) for v in vals.values())
            dg = max(v[1] for v in vals.values())
            nc, per = contacts(m, d, tool_ids)
            ftot = sum(per.values())
            if i % rec_every == 0:
                worst_mm, worst_deg = max(worst_mm, mm), max(worst_deg, dg)
                if False:
                    bw = max(vals, key=lambda b: np.linalg.norm(vals[b][0]))
                    dp = vals[bw][0]
                    ax = info["axis_palm"]
                    along = float(dp.dot(ax))
                    perp = dp - along * ax
                    lost = dict(t=t, body=bw, dp=dp, along=along, perp=perp, deg=vals[bw][1], nc=nc, F=ftot,
                                before=last.copy())
                if csv_rows is not None:
                    row = [run_id, round(t, 2)]
                    for b in tool_bodies:
                        row += [*np.round(vals[b][0], 2), round(float(np.linalg.norm(vals[b][0])), 2), round(vals[b][1], 2)]
                    row += [nc, round(ftot, 3)] + [round(per[f], 3) for f in (*FINGERS, "palm")]
                    if KEY == "tw":
                        row.append(round(float(d.qpos[hinge_adr]), 4))
                    csv_rows.append(row)
                last = dict(t=t, nc=nc, F=ftot, per=per)
            if video_path and i % frame_every == 0:
                bb = m.body("right_base_link").id
                cam.lookat[:] = d.xpos[bb] + d.xmat[bb].reshape(3, 3) @ look_local
                r.update_scene(d, cam)
                img = Image.fromarray(r.render())
                dr = ImageDraw.Draw(img)
                phase = "止める" if t < T_HOLD1 else "手を 上・左・下・右 に10cm" if t < T_HOLD1 + T_MOVE else "止める"
                dr.rectangle([0, H - 96, W, H], fill=(30, 30, 35))
                dr.text((20, H - 88), f"{t:4.1f}秒  {phase}  （{'ピンセット 案1' if KEY == 'tw' else '箸 案3'}・溶接なし）",
                        font=FONT, fill=(255, 212, 0))
                dr.text((20, H - 46), f"ずれ {mm:5.1f} mm  回転 {dg:4.1f} 度  接触 {nc} 個  力 {ftot:5.2f} N",
                        font=FONT, fill=(230, 230, 230))
                ff.stdin.write(np.asarray(img).tobytes())
        if i < steps:
            mujoco.mj_step(m, d)
            if lost is None:                  # 越えた瞬間は 1ms ごとに見る（0.1秒の記録だと回転とずれが同時に越えて見分けられない）
                mujoco.mj_kinematics(m, d)
                vals = {}
                for b in tool_bodies:
                    com, Rt = rel_pose(m, d, b)
                    vals[b] = ((com - ref[b][0]) * 1000, rot_deg(ref[b][1], Rt))
                mm = max(np.linalg.norm(v[0]) for v in vals.values())
                dg = max(v[1] for v in vals.values())
                if mm > LIMIT_MM or dg > LIMIT_DEG:
                    bw = max(vals, key=lambda b: max(np.linalg.norm(vals[b][0]) / LIMIT_MM, vals[b][1] / LIMIT_DEG))
                    dp = vals[bw][0]
                    ax = info["axis_palm"]
                    along = float(dp.dot(ax))
                    nc, per = contacts(m, d, tool_ids)
                    lost = dict(t=t + DT, body=bw, dp=dp, along=along, perp=dp - along * ax, deg=vals[bw][1], nc=nc,
                                F=sum(per.values()), per=per, before=last.copy())
    if video_path:
        ff.stdin.close(); ff.wait(); r.close()
    mujoco.mj_forward(m, d)
    fin = {}
    for b in tool_bodies:
        com, Rt = rel_pose(m, d, b)
        fin[b] = (float(np.linalg.norm((com - ref[b][0]) * 1000)), rot_deg(ref[b][1], Rt))
    fmm = max(v[0] for v in fin.values()); fdg = max(v[1] for v in fin.values())
    held = fmm <= LIMIT_MM and fdg <= LIMIT_DEG
    return dict(held=held, final_mm=fmm, final_deg=fdg, worst_mm=worst_mm, worst_deg=worst_deg, lost=lost, gap0=gap0,
                last=last)


def VIDEO(tag):
    return HERE / (f"grip_hold_{TOOL}.mp4" if tag == "base" else f"grip_hold_{TOOL}_{tag}.mp4")


def describe_lost(L):
    if L is None:
        return "抜けず"
    kind = ("回転して抜ける" if L["deg"] > LIMIT_DEG else
            "軸方向に滑る（指の間を抜ける）" if abs(L["along"]) >= np.linalg.norm(L["perp"]) else
            "軸と直交に押し出される／落ちる")
    p = L["perp"]
    b = L["before"]
    return (f"{L['t']:.3f}秒に越えた（{L['body']}）: {kind} ずれ(手のひら x,y,z) {np.round(L['dp'], 1)}mm "
            f"軸方向 {L['along']:+.1f}mm 直交 {np.linalg.norm(p):.1f}mm 回転 {L['deg']:.1f}度 "
            f"接触 {L['nc']}個・{L['F']:.2f}N（" + " ".join(f"{f}{v:.2f}" for f, v in L['per'].items() if v) + f"）（直前の0.1秒刻み {b.get('nc', '-')}個・{b.get('F', 0):.2f}N）")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    vid = (sys.argv[3] == "1") if len(sys.argv) > 3 else True
    m, tool_bodies, free_bodies, info = compile_with_key()
    traj = arm_traj(m)
    print(f"[{TOOL}] 案{VARIANT}  KP {KP} PRESS {PRESS} MU {MU} impratio {IMPRATIO} noslip {NOSLIP} solimp {SOLIMP[:3]} "
          f"solref {SOLREF} dt {DT}  pair {m.npair} 組  equality {m.neq} 本  actuator {m.nu}")
    print(f"  道具の質量 {[round(float(m.body_subtreemass[m.body(b).id]) * 1000, 1) for b in free_bodies]} g  重力 {m.opt.gravity}  "
          f"weld {sum(m.eq_type == mujoco.mjtEq.mjEQ_WELD)} 本  mocap {m.nmocap}")
    if KEY == "chop":
        print(f"  箸の細い部分 {info['tip_len'] * 1000:.0f}mm（太い部分どうしのすき間 {info['thick_gap'] * 1000:.2f}mm）")
    else:
        print(f"  ピンセットの開き {np.degrees(info['open_ang']):.1f}度  ばね {HINGE_K} N·m/rad")
    print(f"  腕の狙いからのずれ（逆運動学・最大）{traj[4] * 1000:.2f}mm")
    rng = np.random.default_rng(2026)
    rows, res = [], []
    for k in range(n):
        vp = VIDEO(tag) if (vid and k == 0) else None
        r = run(m, traj, tool_bodies, free_bodies, info, rng, vp, rows, k + 1)
        res.append(r)
        lf = r["last"]
        print(f"{tag} {TOOL} #{k + 1} {'保持' if r['held'] else '抜けた'} 最後 {r['final_mm']:.1f}mm/{r['final_deg']:.1f}度 "
              f"最大 {r['worst_mm']:.1f}mm/{r['worst_deg']:.1f}度 初期の最短距離 {r['gap0']:.2f}mm "
              f"最後の接触 {lf['nc']}個・{lf['F']:.2f}N（" + " ".join(f"{f}{v:.2f}" for f, v in lf['per'].items()) + "）| "
              + describe_lost(r["lost"]), flush=True)
    head = ["run", "t"]
    for b in tool_bodies:
        head += [f"{b}_dx_mm", f"{b}_dy_mm", f"{b}_dz_mm", f"{b}_d_mm", f"{b}_rot_deg"]
    head += ["n_contact", "F_total_N"] + [f"F_{f}_N" for f in (*FINGERS, "palm")] + (["tw_hinge_rad"] if KEY == "tw" else [])
    csv = HERE / f"grip_hold_{TOOL}_{tag}.csv"
    csv.write_text(",".join(head) + "\n" + "\n".join(",".join(str(x) for x in row) for row in rows) + "\n")
    print(f"記録: {csv}")
    if vid:
        print(f"動画: {VIDEO(tag)}")
    print(f"SUMMARY {TOOL} 保持 {sum(r['held'] for r in res)}/{n} ずれ最大 "
          + ",".join(f"{r['worst_mm']:.0f}" if r['worst_mm'] < 100 else "落下" for r in res) + " mm 回転最大 "
          + ",".join(f"{r['worst_deg']:.0f}" for r in res) + " 度")


if __name__ == "__main__":
    main()
