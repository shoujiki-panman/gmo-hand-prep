"""Phase 3 の段2：箸で豆を1粒つまんで持ち上げる（物理あり）。

作り方は SIGGRAPH 2022「Learning to Use Chopsticks in Diverse Gripping Styles」の計画段階の扱いに合わせた:
下の箸は手に固定し、上の箸は人差し指の付け根を支点に開閉する「2本爪」として動かす。
- 箸と豆には物理がある（ぶつかる・こすれる・重さがかかる）。箸は手の位置に「溶接」して引っぱる（mocap＋weld）
- G1 の体は見た目だけ（当たり判定なし）。腕は毎回の関節角度を直接入れて動かす
- 腕の角度は「豆の上」「豆の高さ」「持ち上げ」の3か所を乱数で探しておき、その間をなめらかにつなぐ
合格の目安：豆を皿から 2cm 以上持ち上げ、1秒保つ。

  .venv/bin/python chopsticks_pick.py   # 結果を表示し、chopsticks_pick.mp4 を作る
"""

import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import chopsticks_look as cl
import g1_demo_video as g1

HERE = Path(__file__).parent
OUT = HERE / f"chopsticks_pick_{os.environ.get('BEAN_D_MM', '9')}mm.mp4"
DEBUG = __import__("os").environ.get("DEBUG") == "1"
DEBUG_TIMES = (2.0, 2.6, 2.8, 3.0, 3.4, 4.0)
FONT = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", 28)

import os
BEAN_R = float(os.environ.get("BEAN_D_MM", "9")) / 2000   # 豆（球）の半径。BEAN_D_MM=20 で直径2cm の球（事例の大きさ）
BEAN_MASS = 0.0004 * (BEAN_R / 0.0045) ** 3 * float(os.environ.get("BEAN_MASS_SCALE", "1.0"))   # 直径9mm で 0.4g、体積に比例。BEAN_MASS_SCALE で質量だけ倍率（試験10・既定 1.0）
# 箸の場面の皿の位置。箸は長いので、ピンセット用に手前へ寄せた皿（g1.PLATES）だと腕が合わない（10/14）
CHOP_PLATES = [(0.45, -0.20), (0.45, 0.0)]
PLATE_XY = np.array(CHOP_PLATES[0])
PLATE_TOP = g1.TABLE_TOP + 0.008
BEAN = np.array([*PLATE_XY, PLATE_TOP + BEAN_R])
# 上の箸の角度（ラジアン）。0 が見た目のときの形。先のすき間は 角度0で6.6mm、1ラジアンあたり約203mm 開く（実測）
CLOSE = -0.03
# 箸の先を細くする（10/15）：箸1本＝根元側の太いカプセル（直径 8mm）＋先側の細いカプセル（先 3cm・直径 2.5mm）の2段。
# 本物の箸の先は 2〜3mm（目安・出典なし）。中心線（後ろの端〜先端）は今までと同じなので、溶接・ちょうつがい・腕の計算はそのまま
TIP_LEN, TIP_RADIUS = 0.03, 0.00125
# 開く角度は「球の直径＋箸の太さ＋4mm の余裕」が入るように決める。箸の太さは、はさむ点がある先の細い部分（直径 2.5mm）で数える
# （太い 8mm のままだと片側 2.75mm 開きすぎ、20mm の球が 0/5＝持ち上げてから落とした。10/15 実測）
OPEN = (2 * BEAN_R + 2 * TIP_RADIUS + 0.004 - 0.0066) / 0.203
GRIP_BACK = 0.006   # 豆をはさむ点（細い部分 0〜30mm の中）＝箸の先端から 6mm 手前（先端ではさむと、丸い先で豆が前へ押し出された。10/14 実測）
STICK_TILT_DEG = 12  # 箸の下向きの角度（17度だと先が皿に当たった）
GRIP_KP = float(os.environ.get("GRIP_KP", "1.0"))   # N·m/rad。5 だと強すぎて2cm の球がはじき出された。1 で持ち上げて保てた（10/14 実測）
DOWN_DZ = float(os.environ.get("DOWN_DZ_MM", "0")) / 1000   # はさむ高さ down_z の上乗せ（m）。既定 0＝球の中心（往復3回目・10/15）。環境変数で上書き可
# はさむ力（往復2回目・10/15）：閉じるときは角度ではなく力で指定する。はさむ点で PINCH_N（N）になる一定トルクを出す
# （トルク＝PINCH_N×ちょうつがい支点からはさむ点までの距離）。開くときは今までどおり角度指定（GRIP_KP の位置制御）
PINCH_N = float(os.environ.get("PINCH_N", "0.3"))
# 当たりの硬さ（往復3回目・10/15）：豆・皿・箸の先の細い部分。MuJoCo Menagerie Robotiq と同値（往復2回目は 0.005、皿は既定で球が 2mm 沈んだ）
# 机（平面・箱）は既定のまま。solimp の後ろ2つは MuJoCo 既定
HARD_SOLREF, HARD_SOLIMP = [0.004, 1], [0.95, 0.99, 0.001, 0.5, 2]
# 試験5・6（10/15）：豆の滑り摩擦（friction の1つ目だけ）と形。既定は今までどおり 1.0・球
BALL_MU = float(os.environ.get("BALL_MU", "1.0"))
BALL_SHAPE = os.environ.get("BALL_SHAPE", "sphere")   # "sphere"／"ellipsoid"（半径 [r, r, 0.8r]＝上下につぶれた形。質量は同じ）
assert BALL_SHAPE in ("sphere", "ellipsoid"), BALL_SHAPE
# 試験8（10/16）：TRANSFER=1 のときだけ、持ち上げたあと皿2（CHOP_PLATES[1]）へ運んで離す。無いときは今までと同じ
TRANSFER = os.environ.get("TRANSFER") == "1"
# 試験8の測り直し（10/16 本人OK）：豆の接触の次元。既定 4（転がり抵抗なし）。6 で friction の3つ目（転がり 0.001）が効く
BALL_CONDIM = int(os.environ.get("BALL_CONDIM", "6"))   # 10/16 本人OKで既定を 6 に（4 だと皿の上で球が止まらない。実測）
# TRANSFER のとき、終わりを T_HOLD 秒だけ延ばして豆が止まるかを見る（既定 0＝今までどおり 8.0秒）
T_HOLD = float(os.environ.get("T_HOLD", "0"))
NOSLIP = int(os.environ.get("NOSLIP", "2"))   # noslip_iterations（MuJoCo 公式「elliptic＋impratio で足りなければ noslip 1〜3」）
# 手のひらの向き（10/16・passline-hand-look.md）：本物の箸の持ち方（親指が上・手のひらは体の内側でやや下向き）に寄せる弱い項。
# 手のひらの法線＝right_base_link の +x（指が曲がる側。手の座標で指は +z・親指は +y、実測）。
# 体の内側：右手は y≈−0.26、体の中心は y=0 なので内側は +y（合格ラインの「−y」は親指が下になる向きなので +y にした。PALM_REF で変えられる）
# PALM_W=0 で今までと同じ解（残差に 0 が並ぶだけ）
# 既定は 0（今までどおり）。0.5 にすると腕の狙いから最大 105mm ずれて移せなかった（10/16 実測）。理由は STATUS「手の向きの修正（10/16）」
PALM_W = float(os.environ.get("PALM_W", "0"))


def palm_err(n, u):
    """手のひらの法線 n と基準の差を、下の箸の軸 u に直交する面の中だけで比べる（3つ）。
    箸の向きは別の項で決めているので、ここでは箸の軸まわりの回り具合だけを寄せる
    （そのまま n−基準 を入れると箸の向きの項とけんかして腕の狙いから最大 235mm ずれ、移せなかった。10/16 実測）"""
    ref = palm_ref()
    a, r = n - n.dot(u) * u, ref - ref.dot(u) * u
    return a / (np.linalg.norm(a) + 1e-9) - r / (np.linalg.norm(r) + 1e-9)


def palm_ref():
    v = np.array([float(x) for x in os.environ.get("PALM_REF", "0,1,-0.5").split(",")])   # やや下向き＝約27度
    return v / np.linalg.norm(v)


ARM_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
              "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
              "right_wrist_yaw_joint"]


def rot(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis, angle)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def frame_quat(direction):
    """z 軸が direction を向く姿勢（クォータニオン）。"""
    z = direction / np.linalg.norm(direction)
    x = np.cross([0, 0, 1], z) if abs(z[2]) < 0.9 else np.cross([1, 0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return q


class Sticks:
    """手のひら（right_base_link）から見た2本の箸。上の箸は開閉角で向きが変わる。"""

    def __init__(self):
        model, data = g1.build()
        cl.pose(model, data, arm=False)
        lines = cl.chopstick_lines(model, data)
        b = model.body("right_base_link").id
        x, r = data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy()
        loc = lambda p: r.T @ (p - x)
        self.up_rear, self.up_tip = loc(lines["upper"][0]), loc(lines["upper"][1])
        self.lo_rear, self.lo_tip = loc(lines["lower"][0]), loc(lines["lower"][1])
        u = (self.up_tip - self.up_rear) / np.linalg.norm(self.up_tip - self.up_rear)
        self.pivot = self.up_rear + cl.BEHIND * u                   # 人差し指の付け根
        d = self.up_tip - self.lo_tip
        n = d - d.dot(u) * u
        n /= np.linalg.norm(n)
        self.axis = np.cross(u, n)                                   # +角度で下の箸から離れる
        grip_pt = self.up_tip - GRIP_BACK * u                         # 上の箸のはさむ点
        self.grip_arm = float(np.linalg.norm(grip_pt - self.pivot))  # 支点からはさむ点までの距離（m）

    def local(self, theta):
        r = rot(self.axis, theta)
        up = [self.pivot + r @ (p - self.pivot) for p in (self.up_rear, self.up_tip)]
        return {"upper": up, "lower": [self.lo_rear, self.lo_tip]}

    def world(self, data, model, theta):
        b = model.body("right_base_link").id
        x, r = data.xpos[b], data.xmat[b].reshape(3, 3)
        return {k: [x + r @ p for p in v] for k, v in self.local(theta).items()}


def add_stick(body, rear, tip, mass, name):
    """箸1本を2段のカプセルで body に付ける（rear・tip は body の座標）。重さは太い部分に置く。"""
    d = (tip - rear) / np.linalg.norm(tip - rear)
    mid = tip - TIP_LEN * d                                    # 太い部分と細い部分の境目
    kw = dict(friction=[1.0, 0.01, 0.001], condim=4, rgba=[0.80, 0.62, 0.38, 1])
    body.add_geom(name=f"{name}_thick", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[cl.RADIUS, 0, 0],
                  fromto=[*rear, *mid], mass=mass, **kw)
    # 先の細い部分：硬い当たり＋condim 4・priority 1（相手との混ぜ方でこちらの値を優先。往復3回目）
    body.add_geom(name=f"{name}_tip", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[TIP_RADIUS, 0, 0],
                  fromto=[*mid, *tip], mass=mass * 0.002, solref=HARD_SOLREF, solimp=HARD_SOLIMP,
                  priority=1, **kw)


def build(sticks):
    spec = g1.build_spec(props=False, plates=CHOP_PLATES)
    spec.option.timestep = 0.001
    spec.option.impratio = 10
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = NOSLIP
    for g in spec.geoms:                       # G1 は見た目だけ（世界に直接置いた机・皿・床は当たる）
        if g.parent.name != "world":
            g.contype = g.conaffinity = 0
        elif g.type == mujoco.mjtGeom.mjGEOM_CYLINDER:   # 皿（世界の円柱は皿だけ。縁の部品は無い）を硬く。机・床は既定
            g.solref, g.solimp = HARD_SOLREF, HARD_SOLIMP
    w = spec.worldbody
    # 下の箸：手のひらに溶接して引っぱる。溶接のばねは箸の重さに比例して強くなるので、重さは 0.2kg にして
    # 重力は打ち消す（gravcomp）。5g のままだと手から遅れ、豆をはさむ力も出なかった（10/14 実測）
    loc = sticks.local(0.0)
    lo_rear, lo_tip = loc["lower"]
    lo_c, lo_dir = (lo_rear + lo_tip) / 2, lo_tip - lo_rear
    lo_q = frame_quat(lo_dir)
    m9 = np.zeros(9); mujoco.mju_quat2Mat(m9, lo_q); r_lo = m9.reshape(3, 3)
    in_lo = lambda p: r_lo.T @ (p - lo_c)                 # 手のひら座標 → 下の箸の座標
    sticks.lo_c, sticks.r_lo = lo_c, r_lo                 # 下の箸の位置と姿勢（手のひら座標）。動かすときに使う
    lower = w.add_body(name="chop_lower", gravcomp=1)
    lower.add_freejoint()
    # 体の座標の原点は今までどおり箸の中心線の真ん中・z 軸が先の向き（先端＝原点＋z×LENGTH/2。DEBUG の先端計算もこのまま）
    add_stick(lower, np.array([0, 0, -cl.LENGTH / 2]), np.array([0, 0, cl.LENGTH / 2]), 0.2, "chop_lower")
    w.add_body(name="chop_lower_target", mocap=True)
    spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="chop_lower", name2="chop_lower_target", solref=[0.005, 1],
                      data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    # 上の箸：下の箸にちょうつがいでつなぎ、人差し指の付け根を支点に開閉する。開き具合はモーター（狙った角度へ回す）で決める
    up_rear, up_tip = loc["upper"]
    upper = lower.add_body(name="chop_upper", gravcomp=1)
    upper.add_joint(name="chop_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, pos=in_lo(sticks.pivot).tolist(),
                    axis=(r_lo.T @ sticks.axis).tolist(), damping=0.02,
                    range=np.degrees([-0.1, 0.15]).tolist())   # この場面の角度の単位は「度」（ラジアンで書くと動けなくなった）
    add_stick(upper, in_lo(up_rear), in_lo(up_tip), 0.005, "chop_upper")
    grip = spec.add_actuator(name="grip", target="chop_hinge", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    grip.set_to_position(kp=GRIP_KP)
    grip.ctrlrange = [-0.1, 0.15]
    grip.group = 0                              # 開くとき用（角度指定）
    # 閉じるとき用：一定トルクのモーター（ctrl＝トルク N·m、負で閉じる）。使わない側は group ごと止める（drive で切り替え）
    force = spec.add_actuator(name="grip_force", target="chop_hinge", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    force.set_to_motor()
    force.ctrlrange = [-1.0, 1.0]
    force.group = 1
    beans = [BEAN] + [BEAN + [dx, dy, 0] for dx, dy in [(0.04, 0.03), (-0.035, 0.035), (0.03, -0.04)]]
    for i, p in enumerate(beans):
        bean = w.add_body(name=f"bean{i}", pos=p.tolist())
        bean.add_freejoint()
        shape = (dict(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BEAN_R, 0, 0]) if BALL_SHAPE == "sphere" else
                 dict(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[BEAN_R, BEAN_R, 0.8 * BEAN_R]))
        bean.add_geom(**shape, mass=BEAN_MASS,
                      friction=[BALL_MU, 0.01, 0.001], condim=BALL_CONDIM, rgba=[0.62, 0.42, 0.22, 1],
                      solref=HARD_SOLREF, solimp=HARD_SOLIMP,
                      priority=2)   # 試験5やり直し（10/16）：豆を最優先にして、豆との接触は必ず豆の friction・solref・solimp を使う（箸の先は1・皿は0）
    model = spec.compile()
    return model, mujoco.MjData(model)


def set_arm(model, data, q):
    # 指は見た目用の形（cl.SHOW_POSE、10/21）。箸の置き場所は Sticks が cl.HOLD_POSE で決めるので物理は変わらない
    for name, v in {**g1.ARM_POSE, **cl.SHOW_POSE, **cl.ARM_HOLD}.items():
        g1.setj(model, data, name, v)
    for name, v in zip(ARM_JOINTS, q):
        data.qpos[model.joint(name).qposadr[0]] = v


def solve_arm(model, data, sticks, target, theta, start, ref_dir=None, iters=200, axis_term=False):
    """腕の7関節を、少しずつ近づける計算（減衰つき最小二乗法＝逆運動学の定番の解き方）で求める。
    合わせる量（残差）:
      ・箸の先2本の真ん中の位置（target との差、3つ）
      ・2本の先の高さの差（0 にする。上下にずれると豆を皿に押しつけてしまう。10/14 実測）
      ・下の箸の向き（見た目のときの向きに保つ、3つ）
    戻り値は（関節角、位置のずれ mm、先端の高さの差 mm）。"""
    def residual(q):
        set_arm(model, data, q)
        mujoco.mj_kinematics(model, data)
        w = sticks.world(data, model, theta)
        pt = {k: t - GRIP_BACK * (t - r) / np.linalg.norm(t - r) for k, (r, t) in w.items()}
        mid = (pt["upper"] + pt["lower"]) / 2
        u = w["lower"][1] - w["lower"][0]
        u /= np.linalg.norm(u)
        # 上の箸の回転軸（世界の向き）。閉じる動きが水平になるよう、上下方向（箸に直交する成分）に近づける
        b = model.body("right_base_link").id
        ax = data.xmat[b].reshape(3, 3) @ sticks.axis
        up = np.array([0, 0, 1.0]) - u[2] * u
        up /= np.linalg.norm(up)
        up *= np.sign(ax.dot(up)) or 1
        return np.concatenate([(mid - target) * 100,                       # cm 単位
                               [(pt["upper"][2] - pt["lower"][2]) * 100],
                               (u - ref_dir) * 0.5,      # 向きは位置と高さより弱く
                               (ax - up) * (1.0 if axis_term else 0.0),
                               palm_err(data.xmat[b].reshape(3, 3)[:, 0], u) * PALM_W])   # 手のひらの向き（弱い項）

    set_arm(model, data, start)
    mujoco.mj_kinematics(model, data)
    if ref_dir is None:
        w0 = sticks.world(data, model, theta)
        ref_dir = (w0["lower"][1] - w0["lower"][0]) / np.linalg.norm(w0["lower"][1] - w0["lower"][0])
    lo = np.array([model.joint(j).range[0] for j in ARM_JOINTS])
    hi = np.array([model.joint(j).range[1] for j in ARM_JOINTS])
    q = np.array(start, float)
    for _ in range(iters):
        r = residual(q)
        jac = np.zeros((len(r), len(q)))
        for k in range(len(q)):
            dq = np.zeros(len(q)); dq[k] = 1e-4
            jac[:, k] = (residual(q + dq) - r) / 1e-4
        step = np.linalg.solve(jac.T @ jac + 1e-3 * np.eye(len(q)), -jac.T @ r)
        q = np.clip(q + np.clip(step, -0.1, 0.1), lo, hi)
        if np.linalg.norm(step) < 1e-6:
            break
    r = residual(q)
    return q, np.linalg.norm(r[:3]) * 10, abs(r[3]) * 10


REC_HZ = 20          # お手本の記録と AI の指示の間隔（1秒に20回）
OBS_SIZE = 128       # AI に見せる画像の大きさ（ピクセル）


def main(bean=None, hook=None, policy=None, video=True, perturb=None, track=False):
    """bean: 球の置き場所（x, y, z）。hook(obs, action): お手本の記録用に 1秒20回呼ぶ。
    policy(obs) -> (腕7関節の角度, 上の箸の角度): 学習した AI に操作させるとき。video: 動画を作るか。
    perturb: (秒, dx, dy) のとき、その時刻に1回だけ球を横に dx, dy（m）ずらす（途中で見直しているかの試験）。
    track: True のとき、はさむまで（2.0秒まで）は腕を解き直すたびに今の球の位置から狙いを作り直す（見て直し続ける制御）。
    狙い＝今の球の位置＋固定のオフセット（上 5cm・はさむ高さ・持ち上げ 5cm）。時間割は同じ。2.0秒以降は最後に見た位置で固定。"""
    global BEAN
    if bean is not None:
        BEAN = np.array(bean, float)
    sticks = Sticks()
    model, data = build(sticks)
    start = np.array([cl.ARM_HOLD[j] for j in ARM_JOINTS])
    set_arm(model, data, start)
    mujoco.mj_kinematics(model, data)
    w0 = sticks.world(data, model, OPEN)
    ref_dir = (w0["lower"][1] - w0["lower"][0]) / np.linalg.norm(w0["lower"][1] - w0["lower"][0])
    # 箸を寝かせ気味にする（見た目のときは約32度。急だと閉じたとき先が皿に当たり、豆の上を素通りした。10/14 実測）
    horiz = np.array([ref_dir[0], ref_dir[1], 0]) / np.linalg.norm(ref_dir[:2])
    ref_dir = np.cos(np.radians(STICK_TILT_DEG)) * horiz + np.array([0, 0, -np.sin(np.radians(STICK_TILT_DEG))])

    # 豆の中心の高さで横からはさむ。箸の半径(4mm)＋0.5mm だけ皿から浮かせる
    # はさむ高さ＝球の中心（往復3回目。小さい豆の +1.5mm の上乗せはやめた：中心より上ではさむと球を皿へ押し下げ、上の箸が越えて抜けた）
    down_z = BEAN[2] + DOWN_DZ
    above_p = BEAN + [0, 0, 0.05]
    down_p = np.array([*BEAN[:2], down_z])
    lift_p = down_p + [0, 0, 0.05]
    # 時間割（秒）: 箸の先の真ん中の狙い・上の箸の角度・字幕。途中は狙いをまっすぐ動かし、そのつど腕の角度を解く
    plan = [(0.0, above_p, OPEN, "豆の上で箸を開く"),
            (1.0, above_p, OPEN, "豆の高さまで下げる"),
            (2.0, down_p, OPEN, "箸を閉じて豆をはさむ"),
            (2.6, down_p, CLOSE, "はさんだまま持ち上げる"),
            (3.8, lift_p, CLOSE, "持ち上げたまま止める"),
            (4.8, lift_p, CLOSE, "")]
    # 各点の「追従のずれを足す重み」。元の皿の上の点は 1（今までどおり）、皿2の点は 0（絶対位置）
    rel = [1.0] * len(plan)
    if TRANSFER:
        p2 = np.array(CHOP_PLATES[1], float)
        carry_p = np.array([*p2, lift_p[2]])                  # 持ち上げ高さのまま皿2の中心の真上
        low2_p = np.array([*p2, PLATE_TOP + BEAN_R + 0.01])   # 皿の面＋半径＋1cm
        up2_p = low2_p + [0, 0, 0.05]
        T_CARRY = float(os.environ.get("T_CARRY", "1.6"))     # 運ぶ時間（秒）
        t1 = 4.8 + T_CARRY
        plan = plan[:-1] + [(4.8, lift_p, CLOSE, "移す先へ運ぶ"),
                            (t1, carry_p, CLOSE, "下げる"),
                            (t1 + 0.6, low2_p, CLOSE, "箸を開いて離す"),   # ここから角度指定（release_t）
                            (t1 + 1.1, low2_p, OPEN, "箸を上げる"),
                            (t1 + 1.6, up2_p, OPEN, "豆が止まるのを待つ"),
                            (t1 + 1.6 + T_HOLD, up2_p, OPEN, "")]
        if T_HOLD <= 0: plan = plan[:-1]
        rel = [1.0] * 6 + [0.0] * (len(plan) - 6)
        # 離す段階は一定トルクを止め、角度の狙いを CLOSE→OPEN へ 0.5秒かけて動かす（角度指定）。
        # 1回目（10/16）はトルクを一気に切ったため、めり込んでいた接触が戻る力で豆が横へ 46mm/s 弾かれ、皿2の上を転がり続けた（実測）
        release_t = t1 + 0.6
    # 上の箸のちょうつがいの軸を付け直す：豆の高さまで下ろした姿勢で、閉じる動きが水平になる向きにする
    # （軸の向きは作りの上の約束。腕の向きだけで水平にしようとすると、手首の範囲の都合で解けなかった。10/14）
    q_down, _, _ = solve_arm(model, data, sticks, down_p, OPEN, start, ref_dir)
    set_arm(model, data, q_down)
    mujoco.mj_kinematics(model, data)
    u = sticks.world(data, model, OPEN)["lower"]
    u = (u[1] - u[0]) / np.linalg.norm(u[1] - u[0])
    want = np.array([0, 0, 1.0]) - u[2] * u
    want /= np.linalg.norm(want)
    r_base = data.xmat[model.body("right_base_link").id].reshape(3, 3)
    new_axis = r_base.T @ want
    sticks.axis = new_axis * (np.sign(new_axis.dot(sticks.axis)) or 1)
    model, data = build(sticks)
    if DEBUG:
        qd, e, l = solve_arm(model, data, sticks, down_p, OPEN, start, ref_dir)
        set_arm(model, data, qd)
        mujoco.mj_kinematics(model, data)
        for th in (OPEN, 0.0, CLOSE):
            w = sticks.world(data, model, th)
            pt = {k: t - GRIP_BACK * (t - r) / np.linalg.norm(t - r) for k, (r, t) in w.items()}
            print(f"[形] 角度 {th:+.3f}  上のはさむ点-豆 {np.round((pt['upper'] - BEAN) * 1000, 1)}  下 {np.round((pt['lower'] - BEAN) * 1000, 1)}  (逆運動学 {e:.1f}mm/{l:.1f}mm)")
    q_now, c1, l1 = solve_arm(model, data, sticks, above_p, OPEN, start, ref_dir)
    print(f"最初の腕の角度（位置のずれ／先端の高さの差 mm）: {c1:.1f}/{l1:.1f}")
    if DEBUG:
        print(f"[はさむ力] 支点→はさむ点 {sticks.grip_arm * 1000:.1f}mm  PINCH_N {PINCH_N}N → トルク {PINCH_N * sticks.grip_arm:.4f}N·m")
    state = {"q": q_now, "worst": 0.0, "pol_q": q_now, "pol_theta": OPEN}   # AI の最初の指示が出るまでは開始の姿勢

    def at(t):
        for (t0, p0, a0, s0), (t1, p1, a1, _), w0, w1 in zip(plan, plan[1:], rel, rel[1:]):
            if t <= t1:
                k = np.clip((t - t0) / (t1 - t0), 0, 1)
                k = k * k * (3 - 2 * k)                               # なめらかに
                return p0 + (p1 - p0) * k, a0 + (a1 - a0) * k, s0, w0 + (w1 - w0) * k
        return plan[-1][1], plan[-1][2], plan[-2][3], rel[-1]

    seen = {"p": BEAN.copy()}                         # 最後に見た球の位置（track 用）
    TRACK_UNTIL = 2.0                                 # 箸を閉じ始める時刻。ここから先は球を追わない

    def drive(t, solve=True):
        target, theta, label, w = at(t)
        if track:
            if solve and t < TRACK_UNTIL:
                seen["p"] = data.xpos[model.body("bean0").id].copy()   # 今の球の位置を見る
            # 狙いを球のずれの分だけ動かす（オフセットは同じ）。TRANSFER の皿2の点は w=0 で絶対位置
            target = target + (seen["p"] - BEAN) if w == 1.0 else target + (seen["p"] - BEAN) * w
        if policy is not None:                       # AI が操作するときは、AI の指示をそのまま使う
            state["q"], theta = state["pol_q"], state["pol_theta"]
            solve, label = False, "AI が操作中"
        state["theta"] = theta
        if solve:
            # 腕は「開いた形の箸」を基準に解く。閉じる動きで腕が動かないように（閉じると先の真ん中がずれるため）
            state["q"], err, _ = solve_arm(model, data, sticks, target, OPEN, state["q"], ref_dir, iters=6)
            state["worst"] = max(state["worst"], err)
        q = state["q"]
        data.qpos[robot_qpos] = rest[robot_qpos]       # G1 の体は毎回もとの姿勢に戻す（重力で落ちないように）
        set_arm(model, data, q)
        data.qvel[robot_dof] = 0
        mujoco.mj_kinematics(model, data)
        # 下の箸の位置と姿勢を、手のひらの姿勢からそのまま計算する。向きだけ合わせると箸の軸まわりに
        # ねじれ、ちょうつがいでつないだ上の箸が別の場所へ行っていた（10/14 実測の不具合）
        b = model.body("right_base_link").id
        x, r = data.xpos[b], data.xmat[b].reshape(3, 3)
        m = model.body("chop_lower_target").mocapid[0]
        data.mocap_pos[m] = x + r @ sticks.lo_c
        mujoco.mju_mat2Quat(data.mocap_quat[m], (r @ sticks.r_lo).ravel())
        # 閉じる段階（狙いが OPEN より閉じ側）は一定トルク、開く段階は角度指定。使わない側は ctrl 0 で group ごと止める
        closing = theta < OPEN - 1e-9 and not (TRANSFER and t >= release_t)
        model.opt.disableactuator = (1 << 0) if closing else (1 << 1)
        data.ctrl[model.actuator("grip").id] = 0.0 if closing else theta
        data.ctrl[model.actuator("grip_force").id] = -PINCH_N * sticks.grip_arm if closing else 0.0
        state["closing"] = closing
        return label

    # G1 の関節（ちょうつがい形）を全部まとめる。箸と豆は自由に動く関節なので含まない
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
              and model.joint(j).name != "chop_hinge"]
    robot_qpos = np.array([model.jnt_qposadr[j] for j in hinges])
    robot_dof = np.array([model.jnt_dofadr[j] for j in hinges])
    rest = data.qpos.copy()

    # 最初の姿勢で箸を手の位置に置いてから始める
    drive(0.0)
    j = model.body("chop_lower").jntadr[0]
    m = model.body("chop_lower_target").mocapid[0]
    data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 7] = [*data.mocap_pos[m], *data.mocap_quat[m]]
    data.qpos[model.joint("chop_hinge").qposadr[0]] = OPEN
    mujoco.mj_forward(model, data)

    W, H, FPS = 960, 720, 30
    hand_cam = mujoco.MjvCamera()
    hand_cam.lookat[:] = PLATE_XY.tolist() + [PLATE_TOP + 0.02]
    hand_cam.distance, hand_cam.azimuth, hand_cam.elevation = 0.28, 210, -20
    if TRANSFER:                                     # 2つの皿の真ん中を、両方が入る距離から
        hand_cam.lookat[:] = ((PLATE_XY + np.array(CHOP_PLATES[1])) / 2).tolist() + [PLATE_TOP + 0.02]
        hand_cam.distance = 0.55
    # 動画用カメラの向き（環境変数が無いときは上の値のまま）
    hand_cam.azimuth = float(os.environ.get("CAM_AZ", hand_cam.azimuth))
    hand_cam.elevation = float(os.environ.get("CAM_EL", hand_cam.elevation))
    hand_cam.distance = float(os.environ.get("CAM_DIST", hand_cam.distance))
    top_cam = mujoco.MjvCamera()                    # 真上寄りから皿を見るカメラ（AI 用）
    top_cam.lookat[:] = PLATE_XY.tolist() + [PLATE_TOP]
    top_cam.distance, top_cam.azimuth, top_cam.elevation = 0.30, 180, -70
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", str(OUT)],
                          stdin=subprocess.PIPE) if video else None
    bean = model.body("bean0").id
    steps = int(plan[-1][0] / model.opt.timestep)
    frame_every = int(1 / FPS / model.opt.timestep)
    rec_every = int(1 / REC_HZ / model.opt.timestep)
    small = mujoco.Renderer(model, OBS_SIZE, OBS_SIZE) if (hook or policy) else None

    def observe():
        q = np.array([data.qpos[model.joint(j).qposadr[0]] for j in ARM_JOINTS])
        th = data.qpos[model.joint("chop_hinge").qposadr[0]]
        imgs = {}
        for name, cam in (("side", hand_cam), ("top", top_cam)):
            small.update_scene(data, cam)
            imgs[name] = small.render().copy()
        return {"state": np.concatenate([q, [th], data.xpos[bean]]).astype(np.float32), "images": imgs}

    # 乱れ試験：球の自由関節の位置を直接書き換える（速度は0に）
    bean_j = model.body("bean0").jntadr[0]
    bean_qa, bean_da = model.jnt_qposadr[bean_j], model.jnt_dofadr[bean_j]
    perturb_i = int(round(perturb[0] / model.opt.timestep)) if perturb else None

    held_from, max_h = None, 0.0
    if TRANSFER:
        sticks_b = (model.body("chop_lower").id, model.body("chop_upper").id)
        def touching():
            return any(bean in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]) and
                       ({model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]} & set(sticks_b))
                       for c in data.contact[:data.ncon])
        tr = {"lift_max": 0.0, "carry_min": 1.0, "lost_t": None, "lost_xyz": None, "t1": t1}
    r = mujoco.Renderer(model, H, W) if video else None
    if True:
        for i in range(steps):
            t = i * model.opt.timestep
            ob = None
            if i % rec_every == 0 and (hook or policy):
                ob = observe()
                if policy is not None:
                    state["pol_q"], state["pol_theta"] = policy(ob)
            label = drive(t, solve=(i % 10 == 0))     # 腕の角度は 10ms ごとに解き直す
            if perturb_i is not None and i == perturb_i:
                data.qpos[bean_qa:bean_qa + 2] += perturb[1:3]
                data.qvel[bean_da:bean_da + 6] = 0
                mujoco.mj_forward(model, data)
            if perturb_i is not None and perturb_i <= i < perturb_i + int(1.0 / model.opt.timestep):
                label = "球を動かした"                 # 字幕に1秒だけ出す
            if ob is not None and hook is not None:
                hook(ob, np.concatenate([state["q"], [state["theta"]]]).astype(np.float32))
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
                if t >= 2.6 and i % 10 == 0:          # 箸から離れた最初の時刻と場所（10ms ごとに見る）
                    if touching():
                        tr["lost_t"] = None
                    elif tr["lost_t"] is None:
                        tr["lost_t"], tr["lost_xyz"] = t, data.xpos[bean].copy()
            if DEBUG and 2.0 <= t <= (plan[-1][0] if TRANSFER else 4.8) and i % 100 == 0:   # 0.1秒ごと：箸と球の接触の力（法線）とめり込み
                st = []
                for ci, c in enumerate(data.contact[:data.ncon]):
                    b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
                    if bean in (b1, b2):
                        other = model.body(b2 if b1 == bean else b1).name or "world"
                        f = np.zeros(6); mujoco.mj_contactForce(model, data, ci, f)
                        st.append(f"{other}:{f[0]:.3f}N/{-c.dist * 1000:.2f}mm/μ{c.friction[0]:.2f}")   # μ＝接触で実際に使われた滑り摩擦（10/16）
                print(f"[はさむ] t={t:.1f} 角度 {data.qpos[model.joint('chop_hinge').qposadr[0]]:+.4f} "
                      f"{'トルク' if state['closing'] else '角度'} 豆の高さ {h * 1000:.1f}mm 接触(力/めり込み) {st}")
            if DEBUG and any(abs(t - x) < 5e-4 for x in DEBUG_TIMES):
                print(f"t={t:.2f} 上の箸の角度 実 {data.qpos[model.joint('chop_hinge').qposadr[0]]:.3f} 狙い {data.ctrl[model.actuator('grip').id]:.3f} トルク {data.ctrl[model.actuator('grip_force').id]:+.4f}")
                for nm in ("lower",):
                    b = model.body(f"chop_{nm}").id
                    mi = model.body(f"chop_{nm}_target").mocapid[0]
                    z = data.xmat[b].reshape(3, 3)[:, 2]
                    tip_real = data.xpos[b] + z * cl.LENGTH / 2
                    zt = np.zeros(9); mujoco.mju_quat2Mat(zt, data.mocap_quat[mi]); zt = zt.reshape(3, 3)[:, 2]
                    tip_goal = data.mocap_pos[mi] + zt * cl.LENGTH / 2
                    print(f"t={t:.1f} {nm}: 先端のずれ {np.linalg.norm(tip_real - tip_goal) * 1000:.1f}mm  高さ 実 {tip_real[2]:.4f} 狙い {tip_goal[2]:.4f}")
                parts = []
                for ci, c in enumerate(data.contact[:data.ncon]):
                    b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
                    if bean in (b1, b2):
                        other = model.body(b2 if b1 == bean else b1).name or "world"
                        f = np.zeros(6); mujoco.mj_contactForce(model, data, ci, f)
                        parts.append(f"{other}:{f[0]:.3f}N")
                print(f"t={t:.2f} 豆に触れている相手 {parts}")
                up = model.body("chop_upper").id
                ups = sorted({model.body(model.geom_bodyid[c.geom2] if model.geom_bodyid[c.geom1] == up else model.geom_bodyid[c.geom1]).name or "world"
                              for c in data.contact[:data.ncon] if up in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])})
                print(f"t={t:.2f} 上の箸が触れている相手 {ups}  豆の位置 {np.round(data.xpos[bean] * 1000, 1)}")
                for c in data.contact[:data.ncon]:
                    if up in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]):
                        print(f"     上の箸の接触点 {np.round(c.pos * 1000, 1)}  相手の部品 {model.geom(c.geom2 if model.geom_bodyid[c.geom1] == up else c.geom1).name or '(名前なし)'} 種類 {model.geom_type[c.geom2 if model.geom_bodyid[c.geom1] == up else c.geom1]}")
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
        # 持ち上げ成功は 3.8〜4.8秒の最大の高さで判定（最後は皿2に置くので最後の高さでは意味が変わる）
        p2 = np.array(CHOP_PLATES[1], float)
        xy_err = float(np.linalg.norm(data.xpos[bean][:2] - p2))
        on_plate = xy_err <= 0.07 - BEAN_R                                         # (a)
        z_ok = abs(data.xpos[bean][2] - (PLATE_TOP + BEAN_R)) <= 0.003             # (b)
        free = not touching()                                                     # (c)
        lost = tr["lost_t"]
        print(f"移す: 皿2中心から {xy_err * 1000:.1f}mm 高さ {(data.xpos[bean][2] - PLATE_TOP - BEAN_R) * 1000:+.1f}mm 箸と接触 {not free} → {'移せた' if on_plate and z_ok and free else '移せず'}")
        return {"success": bool(tr["lift_max"] >= 0.02), "max_h": float(max_h), "final_h": float(final_h),
                "transferred": bool(on_plate and z_ok and free), "final_xy_err": xy_err * 1000,
                "lift_max": float(tr["lift_max"]), "carry_min": float(tr["carry_min"]), "worst": float(state["worst"]),
                "lost_t": lost, "lost_xyz": None if lost is None else tr["lost_xyz"].tolist(),
                "final_xyz": data.xpos[bean].tolist(),
                "final_speed": float(np.linalg.norm(data.qvel[bean_da:bean_da + 3])) * 1000, "flags": (bool(on_plate), bool(z_ok), bool(free)), "t1": t1}
    return {"success": bool(ok), "max_h": float(max_h), "final_h": float(final_h)}


if __name__ == "__main__":
    # 例: TRACK=1 PERTURB="1.0,0.015,0" BEAN_D_MM=20 python chopsticks_pick.py
    track = os.environ.get("TRACK") == "1"
    perturb = tuple(float(v) for v in os.environ["PERTURB"].split(",")) if os.environ.get("PERTURB") else None
    tag = ("_track" if track else "_notrack") + (
        "_perturb" + "_".join(os.environ["PERTURB"].split(",")) if perturb else "_noperturb")
    OUT = HERE / f"chopsticks_pick_{os.environ.get('BEAN_D_MM', '9')}mm{tag}.mp4"   # 動画名に条件を入れる
    main(track=track, perturb=perturb)
