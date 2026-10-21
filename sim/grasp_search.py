"""Phase 4b 段1b「物理で持ち方を探す」（10/21・学習なし・物理あり）。合格ライン passline-grasp-search.md。

候補をたくさん作り、重力の下で落ちないものだけ残す（ペン回し論文・UniDexGrasp の「候補→物理→ふるい」）。
場面・当たり判定（手の部品×道具の <pair> だけ）・actuator・接触の値・腕を動かす手順は grip_hold.py をそのまま使う
（import して呼ぶだけ。grip_hold.py は変えていない）。weld 0本・mocap 0・重力あり・質量そのまま・dt 1ms。

候補の作り方（2通り）
  (A) 出発点（ピンセット＝案1・箸＝案3）の主関節6つに ±15度、道具の位置に ±8mm（各軸）・向きに ±15度（ランダムな軸のまわり、
      道具の重心が中心）の一様乱数。初期のめり込みが 0.3mm を超える指は、その指の曲げ関節を 0.01rad ずつ開いて 0.3mm 以下にする
      （手のひら・親指の付け根がめり込む候補は捨てて引き直す）。狙いの角度＝角度＋PRESS（grip_hold と同じ指だけ）
  (B) 閉じて握る: 指を開いた形（曲げ関節 0）、親指の付け根の回転は一様乱数。道具を出発点の位置 ±B_POS_MM・向き ±B_ROT_DEG に置き、
      道具をその場に止めたまま（生成のときだけ。毎ステップ位置を上書き）、曲げ関節の狙いを B_SPEED rad/s で閉じ側へ送る。
      指が道具に B_STOP_N 以上の力で触れたら、その指の狙い＝その時の角度＋PRESS で止める。0.15秒なじませた状態を候補にする。
      親指と、ほかの指1本以上が触れなかった候補は捨てて引き直す
ふるい（同じ1回の物理の中で続けて見る。越えたらその場で打ち切り）
  0〜1秒 腕を止める → 1〜3秒 止めたまま → 3〜5秒 手のひらを 上・左・下・右 に10cm（各0.5秒・grip_hold の手順）→ 5〜6秒 止める
  道具の重心の手のひら座標でのずれ 5mm・回転 5度を、毎ステップ（1ms）見る（基準は 0秒＝候補の形）
質の指標（候補ごと）
  初期（0秒）: 手と道具の距離 1mm 以内の組を接触点とみなす（指標を出すためだけの複製モデル。物理には使わない）
  保持中（1秒）: 実際の接触（力で重みづけ）
  - 接触点の数／親指と対向する指の押す向きの角度（親指の押す向きと、いちばん向かい合う指の押す向き。180度に近いほど良い）
  - 道具の重心を水平面に投影した点が、接触点の作る多角形（凸包）の中に入るか・境界からの距離 mm（＋が内側）

  cd ~/Developer/gmo-hand/lerobot && caffeinate -i .venv/bin/python <このフォルダ>/grasp_search.py tweezers 300
  cd ~/Developer/gmo-hand/lerobot && caffeinate -i .venv/bin/python <このフォルダ>/grasp_search.py chopsticks 300
  引数: 道具 1通りあたりの候補数（A・B それぞれこの数）

環境変数（往復2回目は候補の作り方だけ変える）:
  A_DEG=15 A_POS_MM=8 A_ROT_DEG=15 B_POS_MM=12 B_ROT_DEG=20 B_SPEED=1.5 B_STOP_N=0.05 WORKERS=9 SEED=2026 TAG=
"""

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOOL = os.environ.get("GS_TOOL") or (sys.argv[1] if len(sys.argv) > 1 else "tweezers")
os.environ["TOOL"] = TOOL
os.environ["GS_TOOL"] = TOOL

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grip_hold as gh  # noqa: E402  場面・当たり判定・actuator・接触の値（変えずに使う）

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

KEY = gh.KEY
A_DEG = float(os.environ.get("A_DEG", "15"))
A_POS = float(os.environ.get("A_POS_MM", "8")) / 1000
A_ROT = float(os.environ.get("A_ROT_DEG", "15"))
B_POS = float(os.environ.get("B_POS_MM", "12")) / 1000
B_ROT = float(os.environ.get("B_ROT_DEG", "20"))
B_SPEED = float(os.environ.get("B_SPEED", "1.5"))
B_STOP = float(os.environ.get("B_STOP_N", "0.05"))
WORKERS = int(os.environ.get("WORKERS", "9"))
SEED = int(os.environ.get("SEED", "2026"))
TAG = os.environ.get("TAG", "")
PEN_TOL = 0.3          # mm。これ以上のめり込みは開いて直す／捨てる
METRIC_MARGIN = 0.001  # 初期の指標で「接触点」とみなす距離 1mm

# ふるいの時刻（grip_hold の腕の手順を 3秒からに置き換えるだけ）
T1, T3, TMOVE, TEND = 1.0, 3.0, 2.0, 6.0
gh.T_HOLD1, gh.T_MOVE, gh.T_HOLD2 = T3, TMOVE, TEND - T3 - TMOVE
gh.T_END = TEND
FLEX = gh.MAIN[1:]     # 曲げ関節（親指 proximal と4本の指）。MAIN[0] は親指の付け根の回転
FINGER_JOINTS = None   # 指の全関節（qpos を候補として保存する）

FONT = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", 24)
FONT_S = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 19)

W = {}  # ワーカーごとのモデル


# ---------------------------------------------------------------- 準備
def setup():
    m, tool_bodies, free_bodies, info = gh.compile_with_key()
    traj = gh.arm_traj(m)
    mm = copy.deepcopy(m)                      # 指標だけに使う複製（距離 1mm 以内を接触として拾う）
    mm.pair_margin[:] = METRIC_MARGIN
    mm.pair_gap[:] = METRIC_MARGIN
    tool_ids = {m.body(b).id for b in free_bodies + (["tool_b"] if KEY == "tw" else [])}
    fj = [j for j in range(m.njnt) if m.joint(j).name.startswith("right_") and any(f in m.joint(j).name for f in gh.FINGERS)]
    hand_dof = set(fj) | {j for j in range(m.njnt) if m.joint(j).name.endswith("_free") or m.joint(j).name == "tw_hinge"}
    kin = [j for j in range(m.njnt) if j not in hand_dof]
    d0 = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d0, 0)
    mujoco.mj_forward(m, d0)
    # 道具の重心（手のひら座標）＝ 向きを回す中心
    pb = m.body("right_base_link").id
    x, R = d0.xpos[pb], d0.xmat[pb].reshape(3, 3)
    masses = [m.body_subtreemass[m.body(b).id] for b in free_bodies]
    coms = [R.T @ (d0.subtree_com[m.body(b).id] - x) for b in free_bodies]
    center = sum(w * c for w, c in zip(masses, coms)) / sum(masses)
    # 道具の当たり判定 geom と、手の当たり判定 geom（finger ごと）
    tool_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in tool_ids and m.geom_contype[g] == 0}
    hand_geoms = sorted({int(m.pair_geom1[p]) for p in range(m.npair)})
    hand_geom_finger = {g: gh.finger_of(m.body(m.geom_bodyid[g]).name) for g in hand_geoms}
    W.update(m=m, mm=mm, dm=mujoco.MjData(mm), tool_bodies=tool_bodies, free_bodies=free_bodies, info=info, traj=traj,
             tool_ids=tool_ids, finger_jnts=fj, kin_q=np.array([m.jnt_qposadr[j] for j in kin]),
             kin_v=np.array([m.jnt_dofadr[j] for j in kin]), rest=d0.qpos.copy(), center=center,
             hand_geoms=hand_geoms, hand_geom_finger=hand_geom_finger, key_ctrl=m.key_ctrl[0].copy(),
             press=np.array([gh.PRESS if n in gh.PRESS_J else 0.0 for n in gh.MAIN]),
             hinge=(m.jnt_qposadr[m.joint("tw_hinge").id] if KEY == "tw" else None))
    return W


def palm(m, d):
    b = m.body("right_base_link").id
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy(), d.xquat[b].copy()


def rotvec_quat(rv):
    q = np.array([1.0, 0, 0, 0])
    a = np.linalg.norm(rv)
    if a > 1e-12:
        mujoco.mju_axisAngle2Quat(q, rv / a, a)
    return q


def quat_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def tool_rel_from_delta(t, rv):
    """手のひら座標での道具のずらし（重心のまわりに rv 回して t 動かす）→ 道具の body の手のひら座標での位置と向き。
    body の原点は手のひらの原点（grip_hold は geom を手のひら座標で作り、body を手のひらに重ねて置いている）。"""
    c = W["center"]
    Rd = quat_mat(rotvec_quat(rv))
    return (c - Rd @ c + t), rotvec_quat(rv)


def apply_state(d, cand):
    """候補（指の関節・道具の手のひら座標での位置と向き・狙いの角度）を data に入れる。腕・体は keyframe のまま。"""
    m = W["m"]
    d.qpos[:] = W["rest"]
    d.qvel[:] = 0
    d.act[:] = 0
    for j, v in zip(W["finger_jnts"], cand["finger_q"]):
        d.qpos[m.jnt_qposadr[j]] = v
    mujoco.mj_kinematics(m, d)
    x, R, qp = palm(m, d)
    p, q = np.array(cand["tool_pos"]), np.array(cand["tool_quat"])
    for fb in W["free_bodies"]:
        a = m.jnt_qposadr[m.body(fb).jntadr[0]]
        qw = np.zeros(4)
        mujoco.mju_mulQuat(qw, qp, q)
        d.qpos[a:a + 3] = x + R @ p
        d.qpos[a + 3:a + 7] = qw
    if W["hinge"] is not None:
        d.qpos[W["hinge"]] = cand.get("hinge", 0.0)
    d.ctrl[:] = cand["ctrl"]
    mujoco.mj_forward(m, d)


def finger_q_from_main(m, d, main):
    for n, v in zip(gh.MAIN, main):
        gh.g1.setj(m, d, n, v)
    return [float(d.qpos[m.jnt_qposadr[j]]) for j in W["finger_jnts"]]


def pen_by_finger(m, d):
    """手の当たり判定ごとの道具までの最短距離（mm、負＝めり込み）を、部位（親指・人差し指…・手のひら）ごとの最小で。"""
    out = {}
    ft = np.zeros(6)
    for p in range(m.npair):
        g1_, g2_ = m.pair_geom1[p], m.pair_geom2[p]
        dist = mujoco.mj_geomDistance(m, d, g1_, g2_, 0.05, ft) * 1000
        f = W["hand_geom_finger"][int(g1_)]
        out[f] = min(out.get(f, 1e9), dist)
    return out


# ---------------------------------------------------------------- 指標
def contact_list(m, d, weighted):
    """[(finger, pos, 押す向き（道具にかかる向き）, 道具の body, 力)]"""
    out = []
    f6 = np.zeros(6)
    for i, c in enumerate(d.contact[:d.ncon]):
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in W["tool_ids"]) == (b2 in W["tool_ids"]):
            continue
        n = c.frame[:3].copy()          # geom1 → geom2
        if b1 in W["tool_ids"]:
            n, hand_b, tool_b = -n, b2, b1
        else:
            hand_b, tool_b = b1, b2
        fn = 1.0
        if weighted:
            mujoco.mj_contactForce(m, d, i, f6)
            fn = float(f6[0])
            if fn <= 1e-6:
                continue
        out.append((gh.finger_of(m.body(hand_b).name), c.pos.copy(), n, int(tool_b), fn))
    return out


def hull2d(pts):
    pts = sorted(set(map(tuple, np.round(pts, 7))))
    if len(pts) < 3:
        return [np.array(p) for p in pts]
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return [np.array(p) for p in lo[:-1] + up[:-1]]


def seg_pt(p, a, b):
    ab = b - a
    t = np.clip((p - a).dot(ab) / max(ab.dot(ab), 1e-18), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def margin_in_hull(p, hull):
    """多角形の境界までの距離 m（＋内側・−外側）。点が2つ以下なら −（線や点では支えられない）。"""
    if len(hull) == 0:
        return -1.0
    if len(hull) == 1:
        return -float(np.linalg.norm(p - hull[0]))
    if len(hull) == 2:
        return -seg_pt(p, hull[0], hull[1])
    dmin = min(seg_pt(p, hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull)))
    inside = all(((hull[(i + 1) % len(hull)] - hull[i])[0] * (p - hull[i])[1]
                  - (hull[(i + 1) % len(hull)] - hull[i])[1] * (p - hull[i])[0]) >= 0 for i in range(len(hull)))
    return dmin if inside else -dmin


def metrics(m, d, weighted):
    cl = contact_list(m, d, weighted)
    per = {}
    for f, p, n, tb, fn in cl:
        per.setdefault(f, []).append(n * fn)
    dirs = {f: sum(v) / max(np.linalg.norm(sum(v)), 1e-12) for f, v in per.items()}
    opp, opp_f = 0.0, None
    if "thumb" in dirs:
        for f, v in dirs.items():
            if f in ("thumb", "palm"):   # 対向する「指」だけ（手のひらは数えない）
                continue
            a = float(np.degrees(np.arccos(np.clip(dirs["thumb"].dot(v), -1, 1))))
            if a > opp:
                opp, opp_f = a, f
    # 重心の投影（水平面）が接触多角形に入るか。ピンセットは本体＋腕B（1つの道具）、箸は1本ずつ見て悪い方
    groups = ([("tool_a", W["tool_ids"])] if KEY == "tw" else [(b, {m.body(b).id}) for b in W["free_bodies"]])
    margins = []
    for b, ids in groups:
        com = d.subtree_com[m.body(b).id][:2]
        pts = [p[:2] for f, p, n, tb, fn in cl if tb in ids]
        margins.append(margin_in_hull(com, hull2d(pts)))
    mg = min(margins)
    if mg <= -0.999:          # 接触が1つも無い道具がある
        mg = float("nan")
    return dict(n_contact=len(cl), fingers=sorted(per), opposition_deg=round(opp, 1), opposed_finger=opp_f,
                com_inside=bool(mg > 0), com_margin_mm=(None if mg != mg else round(mg * 1000, 2)))


def init_metrics(cand):
    mm, dm = W["mm"], W["dm"]
    dm.qpos[:] = 0
    # 同じ状態を複製モデルに入れる（qpos だけで決まる）
    d = mujoco.MjData(W["m"])
    apply_state(d, cand)
    dm.qpos[:] = d.qpos
    mujoco.mj_forward(mm, dm)
    return metrics(mm, dm, weighted=False)


# ---------------------------------------------------------------- 候補を作る
def rear_points(cand):
    """道具の後ろ端（手のひら座標 mm）。"""
    p, R = np.array(cand["tool_pos"]), quat_mat(np.array(cand["tool_quat"]))
    L = gh.DESIGN["lines"]
    keys = ["A"] if KEY == "tw" else ["upper", "lower"]
    return {k: [round(float(v), 1) for v in (p + R @ np.array(L[k]["rear"])) * 1000] for k in keys}


def make_A(seed):
    m = W["m"]
    rng = np.random.default_rng([SEED, 1, seed])
    base = np.array([gh.DESIGN["hand_pose"][n] for n in gh.MAIN])
    lo = np.array([m.jnt_range[m.joint(n).id][0] for n in gh.MAIN])
    hi = np.array([m.jnt_range[m.joint(n).id][1] for n in gh.MAIN])
    main = np.clip(base + np.radians(rng.uniform(-A_DEG, A_DEG, 6)), lo, hi)
    t = rng.uniform(-A_POS, A_POS, 3)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    rv = ax * np.radians(rng.uniform(-A_ROT, A_ROT))
    p, q = tool_rel_from_delta(t, rv)
    d = mujoco.MjData(m)
    cand = dict(method="A", seed=int(seed), main_sampled=main.tolist(), delta_mm=(t * 1000).tolist(),
                delta_rot_deg=float(np.degrees(np.linalg.norm(rv))), tool_pos=p.tolist(), tool_quat=q.tolist(), hinge=0.0)
    backed = {}
    for _ in range(200):
        cand["finger_q"] = finger_q_from_main(m, d, main)
        cand["ctrl"] = (main + W["press"]).tolist()
        apply_state(d, cand)
        pen = pen_by_finger(m, d)
        if pen.get("palm", 1e9) < -PEN_TOL:
            return None, "手のひらがめり込む"
        bad = [f for f in gh.FINGERS if pen.get(f, 1e9) < -PEN_TOL]
        if not bad:
            break
        for f in bad:
            k = 1 if f == "thumb" else gh.FINGERS.index(f) + 1
            if main[k] <= lo[k] + 1e-9:
                return None, f"{f}が開ききってもめり込む"
            main[k] = max(lo[k], main[k] - 0.01)
            backed[f] = backed.get(f, 0) + 1
    else:
        return None, "めり込みが直らない"
    cand["main"] = main.tolist()
    cand["backed_off_rad"] = {f: round(v * 0.01, 2) for f, v in backed.items()}
    cand["min_gap_mm"] = round(min(pen.values()), 2)
    return cand, None


def make_B(seed):
    m = W["m"]
    rng = np.random.default_rng([SEED, 2, seed])
    lo = np.array([m.jnt_range[m.joint(n).id][0] for n in gh.MAIN])
    hi = np.array([m.jnt_range[m.joint(n).id][1] for n in gh.MAIN])
    main = lo.copy()
    main[0] = rng.uniform(lo[0], hi[0])          # 親指の付け根の回転だけ乱数（曲げ関節は開いた形 0）
    t = rng.uniform(-B_POS, B_POS, 3)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    rv = ax * np.radians(rng.uniform(-B_ROT, B_ROT))
    p, q = tool_rel_from_delta(t, rv)
    d = mujoco.MjData(m)
    cand = dict(method="B", seed=int(seed), thumb_rot=float(main[0]), delta_mm=(t * 1000).tolist(),
                delta_rot_deg=float(np.degrees(np.linalg.norm(rv))), tool_pos=p.tolist(), tool_quat=q.tolist(), hinge=0.0)
    cand["finger_q"] = finger_q_from_main(m, d, main)
    cand["ctrl"] = main.tolist()
    apply_state(d, cand)
    pen = pen_by_finger(m, d)
    if min(pen.values()) < -PEN_TOL:
        return None, "開いた手にめり込む"
    # 閉じる（道具はその場に止める＝生成のときだけ。評価では止めない）
    tool_q = [(m.jnt_qposadr[m.body(fb).jntadr[0]], m.jnt_dofadr[m.body(fb).jntadr[0]]) for fb in W["free_bodies"]]
    tool_q0 = [d.qpos[a:a + 7].copy() for a, _ in tool_q]
    stopped = {}
    ctrl = main.copy()
    madr = [m.jnt_qposadr[m.joint(n).id] for n in gh.MAIN]
    t_close = (hi[1:] - lo[1:]).max() / B_SPEED + 0.3
    steps = int(round(t_close / gh.DT))
    settle = None
    for i in range(steps + int(0.15 / gh.DT)):
        if settle is None:
            touch = {}
            for f, pp, n, tb, fn in contact_list(m, d, weighted=True):
                touch[f] = touch.get(f, 0.0) + fn
            for k, jn in enumerate(gh.MAIN):
                if k == 0:
                    continue
                f = "thumb" if k == 1 else gh.FINGERS[k - 1]
                if k in stopped:
                    continue
                if touch.get(f, 0.0) >= B_STOP:
                    stopped[k] = float(d.qpos[madr[k]])
                    ctrl[k] = min(hi[k], stopped[k] + (gh.PRESS if jn in gh.PRESS_J else 0.0))
                else:
                    ctrl[k] = min(hi[k], ctrl[k] + B_SPEED * gh.DT)
            if len(stopped) == 5 or all(ctrl[k] >= hi[k] - 1e-9 or k in stopped for k in range(1, 6)):
                settle = i
            d.ctrl[:] = ctrl
        elif i - settle >= int(0.15 / gh.DT):
            break
        mujoco.mj_step(m, d)
        for (a, v), q0 in zip(tool_q, tool_q0):
            d.qpos[a:a + 7] = q0
            d.qvel[v:v + 6] = 0
        if W["hinge"] is not None:
            d.qpos[W["hinge"]] = 0.0
            d.qvel[m.jnt_dofadr[m.joint("tw_hinge").id]] = 0
    mujoco.mj_forward(m, d)
    fing = {gh.FINGERS[k - 1] if k > 1 else "thumb": round(v, 3) for k, v in stopped.items()}
    if "thumb" not in fing or len(fing) < 2:
        return None, "閉じても親指＋1本が触れない"
    cand["finger_q"] = [float(d.qpos[m.jnt_qposadr[j]]) for j in W["finger_jnts"]]
    cand["main"] = [float(d.qpos[a]) for a in madr]
    cand["ctrl"] = ctrl.tolist()
    cand["stopped_at"] = fing
    cand["min_gap_mm"] = round(min(pen_by_finger(m, d).values()), 2)
    return cand, None


# ---------------------------------------------------------------- 物理で確かめる
def evaluate(cand, video=None, caption=""):
    m = W["m"]
    ts, Q, qadr, dadr, _ = W["traj"]
    d = mujoco.MjData(m)
    apply_state(d, cand)
    ref = {b: gh.rel_pose(m, d, b) for b in W["tool_bodies"]}
    kin_q, kin_v, rest = W["kin_q"], W["kin_v"], W["rest"]
    steps = int(round(TEND / gh.DT))
    worst = {"1s": [0.0, 0.0], "3s": [0.0, 0.0], "arm": [0.0, 0.0]}
    lost, m1 = None, None
    ff = rd = None
    if video:
        VW, VH, FPS = 960, 720, 30
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}",
                               "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", str(video)], stdin=subprocess.PIPE)
        rd = mujoco.Renderer(m, VH, VW)
        cam = mujoco.MjvCamera()
        x, R, _ = palm(m, d)
        a = R @ (quat_mat(np.array(cand["tool_quat"])) @ W["info"]["axis_palm"])
        h = np.cross(a, [0, 0, 1.0])
        h /= np.linalg.norm(h)
        cam.azimuth = float(np.degrees(np.arctan2(h[1], h[0])))
        cam.elevation, cam.distance = -15, 0.22 if KEY == "tw" else 0.36
        look = np.array(cand["tool_pos"]) + quat_mat(np.array(cand["tool_quat"])) @ W["center"]
        frame_every = int(round(1 / FPS / gh.DT))
    for i in range(steps + 1):
        t = i * gh.DT
        k = min(int(t / 0.01), len(ts) - 2)
        u = (t - ts[k]) / 0.01
        d.qpos[kin_q] = rest[kin_q]
        d.qvel[kin_v] = 0
        d.qpos[qadr] = Q[k] + (Q[k + 1] - Q[k]) * u
        d.qvel[dadr] = (Q[k + 1] - Q[k]) / 0.01
        if video and i % frame_every == 0:
            mujoco.mj_forward(m, d)
            x, R, _ = palm(m, d)
            cam.lookat[:] = x + R @ look
            rd.update_scene(d, cam)
            img = Image.fromarray(rd.render())
            dr = ImageDraw.Draw(img)
            mmv = max(np.linalg.norm((gh.rel_pose(m, d, b)[0] - ref[b][0]) * 1000) for b in W["tool_bodies"])
            dgv = max(gh.rot_deg(ref[b][1], gh.rel_pose(m, d, b)[1]) for b in W["tool_bodies"])
            nc, per = gh.contacts(m, d, W["tool_ids"])
            ph = "止める" if t < T3 else "手を 上・左・下・右 に10cm" if t < T3 + TMOVE else "止める"
            dr.rectangle([0, VH - 96, VW, VH], fill=(30, 30, 35))
            dr.text((20, VH - 88), f"{t:4.1f}秒  {ph}  {caption}", font=FONT, fill=(255, 212, 0))
            dr.text((20, VH - 46), f"ずれ {mmv:5.1f} mm  回転 {dgv:4.1f} 度  接触 {nc} 個  力 {sum(per.values()):5.2f} N",
                    font=FONT, fill=(230, 230, 230))
            ff.stdin.write(np.asarray(img).tobytes())
        if i == int(round(T1 / gh.DT)) and lost is None:
            mujoco.mj_forward(m, d)
            m1 = metrics(m, d, weighted=True)
        if i == steps:
            break
        mujoco.mj_step(m, d)
        if lost is None:
            # mj_step の中で計算した位置（このステップの始め）で判定する
            st = "1s" if t < T1 else "3s" if t < T3 else "arm"
            for b in W["tool_bodies"]:
                com, Rt = gh.rel_pose(m, d, b)
                dp = (com - ref[b][0]) * 1000
                mmv, dgv = float(np.linalg.norm(dp)), gh.rot_deg(ref[b][1], Rt)
                w = worst[st]
                w[0], w[1] = max(w[0], mmv), max(w[1], dgv)
                if mmv > gh.LIMIT_MM or dgv > gh.LIMIT_DEG:
                    ax = ref[b][1] @ W["info"]["axis_palm"] if KEY == "tw" else ref[b][1] @ axis_body(b)
                    along = float(dp.dot(ax))
                    perp = float(np.linalg.norm(dp - along * ax))
                    kind = ("回転" if dgv > gh.LIMIT_DEG else "軸方向に滑る" if abs(along) >= perp else "軸と直交に抜ける")
                    nc, per = gh.contacts(m, d, W["tool_ids"])
                    lost = dict(t=round(t, 3), stage=st, body=b, kind=kind, mm=round(mmv, 2), deg=round(dgv, 2),
                                along_mm=round(along, 2), perp_mm=round(perp, 2), nc=nc,
                                F={f: round(v, 3) for f, v in per.items() if v})
                    break
            if lost is not None and not video:
                break
    if video:
        ff.stdin.close()
        ff.wait()
        rd.close()
    stage = ("0" if lost and lost["stage"] == "1s" else "1s" if lost and lost["stage"] == "3s" else
             "3s" if lost else "arm")
    return dict(passed=stage, lost=lost, worst={k: [round(a, 2), round(b, 2)] for k, (a, b) in worst.items()}, m1=m1)


def axis_body(b):
    L = gh.DESIGN["lines"]["lower" if b == "chop_lo" else "upper"]
    v = np.array(L["tip"]) - np.array(L["rear"])
    return v / np.linalg.norm(v)


def job(arg):
    method, seed = arg
    t0 = time.time()
    cand, why = (make_A if method == "A" else make_B)(seed)
    if cand is None:
        return dict(method=method, seed=seed, rejected=why)
    cand["m0"] = init_metrics(cand)
    cand["rear_mm"] = rear_points(cand)
    cand.update(evaluate(cand))
    cand["sec"] = round(time.time() - t0, 2)
    return cand


def init_worker():
    setup()


# ---------------------------------------------------------------- 撮る
VIEWS = [("甲側", np.array([-1.0, 0.45, 0.0])), ("親指側", np.array([-0.15, 1.0, 0.0])),
         ("斜め（甲・親指・前から）", np.array([-0.6, 0.7, 0.45]))]   # grip_design.py と同じ向き（手のひら座標）


def render_sheet(items, out, title):
    m = W["m"]
    keep = m.geom_rgba.copy()
    hand_b = {b for b in range(m.nbody) if gh.is_hand_body(m.body(b).name) or "tip" in m.body(b).name and m.body(b).name.startswith("right_")}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in hand_b and m.geom_bodyid[g] not in W["tool_ids"]:
            m.geom_rgba[g, 3] = 0
    S = 420
    d = mujoco.MjData(m)
    rows = []
    with mujoco.Renderer(m, S, S) as rd:
        for it in items:
            apply_state(d, it)
            x, R, _ = palm(m, d)
            look = x + R @ (np.array(it["tool_pos"]) + quat_mat(np.array(it["tool_quat"])) @ W["center"])
            imgs = []
            for _, c in VIEWS:
                v = R @ (c / np.linalg.norm(c))
                cam = mujoco.MjvCamera()
                cam.lookat[:] = look
                cam.distance = 0.24 if KEY == "tw" else 0.32
                cam.azimuth = float(np.degrees(np.arctan2(-v[1], -v[0])))
                cam.elevation = float(np.degrees(np.arcsin(np.clip(-v[2], -1, 1))))
                rd.update_scene(d, cam)
                imgs.append(Image.fromarray(rd.render()))
            rows.append((it, imgs))
    m.geom_rgba[:] = keep
    ref = HERE / "reports" / "ref_feix2016_fig4_20_21.png"
    ref = Image.open(ref).convert("RGB") if ref.exists() else None
    head, capH = 150, 100
    sheet = Image.new("RGB", (S * 3 + 330, head + max(1, len(rows)) * (S + capH)), (250, 250, 252))
    dr = ImageDraw.Draw(sheet)
    dr.text((16, 12), title, font=FONT, fill=(20, 20, 30))
    dr.text((16, 46), "列＝甲側・親指側・斜め（候補の 0秒の形）。手のひら座標: +x 手のひら側・+y 親指側・+z 指の向き。物理で確かめた結果つき",
            font=FONT_S, fill=(60, 60, 70))
    for i, (nm, _) in enumerate(VIEWS):
        dr.text((i * S + 16, head - 32), nm, font=FONT, fill=(20, 20, 30))
    if ref is not None:
        half = ref.crop((0, 0, ref.width // 2, ref.height)) if KEY == "chop" else ref.crop((ref.width // 2, 0, ref.width, ref.height))
        half.thumbnail((300, S + head))
        sheet.paste(half, (S * 3 + 15, head))
        dr.text((S * 3 + 15, head + half.height + 6), "見本: Feix ほか 2016\nIEEE THMS 46(1) Fig.4", font=FONT_S, fill=(60, 60, 70))
    for ri, (it, imgs) in enumerate(rows):
        y0 = head + ri * (S + capH)
        for ci, im in enumerate(imgs):
            sheet.paste(im, (ci * S, y0))
        m1 = it.get("m1") or {}
        res = {"arm": "腕を動かしても残った", "3s": "3秒は残った・腕で落ちた", "1s": "1秒は残った・3秒までに落ちた", "0": "1秒以内に落ちた"}[it["passed"]]
        l1 = (f"{ri + 1}位  {it['method']}#{it['seed']}  {res}" + (f"（{it['lost']['t']:.3f}秒・{it['lost']['kind']}）" if it["lost"] else ""))
        l2 = (f"1秒時点: 接触 {m1.get('n_contact', '-')}点・押す向き 親指と{m1.get('opposed_finger', '-')} {m1.get('opposition_deg', '-')}°・"
              f"重心の投影 {'内' if m1.get('com_inside') else '外'}（{m1.get('com_margin_mm', '-')}mm）  0秒: 接触 {it['m0']['n_contact']}点・"
              f"{it['m0']['opposition_deg']}°  指の角度 " + ",".join(f"{v:.2f}" for v in it["main"]) + " rad")
        dr.text((12, y0 + S + 8), l1, font=FONT, fill=(20, 20, 30))
        dr.text((12, y0 + S + 44), l2, font=FONT_S, fill=(20, 20, 30))
    sheet.save(out)


# ---------------------------------------------------------------- 集計
def stats(vals, nd=1):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return "-"
    return f"中央 {np.median(v):.{nd}f}（{np.percentile(v, 10):.{nd}f}〜{np.percentile(v, 90):.{nd}f}）"


def report(res, method):
    rs = [r for r in res if r["method"] == method]
    n = len(rs)
    s1 = [r for r in rs if r["passed"] in ("1s", "3s", "arm")]
    s3 = [r for r in rs if r["passed"] in ("3s", "arm")]
    sj = [r for r in rs if r["passed"] == "arm"]
    print(f"  ({method}) 候補 {n} 1秒残り {len(s1)} 3秒残り {len(s3)} 腕を動かして残り {len(sj)}")
    for nm, grp in (("1秒で落ちた", [r for r in rs if r["passed"] == "0"]), ("1秒は残った", s1), ("腕まで残った", sj)):
        if not grp:
            continue
        kinds = {}
        for r in grp:
            if r["lost"]:
                kinds[r["lost"]["kind"]] = kinds.get(r["lost"]["kind"], 0) + 1
        rk = list(rear_key(grp))
        print(f"    {nm} {len(grp)}: 初期 押す向き {stats([r['m0']['opposition_deg'] for r in grp])}° 接触点 "
              f"{stats([r['m0']['n_contact'] for r in grp])} 重心の投影が内 {sum(r['m0']['com_inside'] for r in grp)}  "
              f"後ろ端 x {stats([v[0] for v in rk])} z {stats([v[2] for v in rk])}mm  "
              f"落ちた時刻 {stats([r['lost']['t'] for r in grp if r['lost']], 3)}秒 落ち方 {kinds}")
    return n, len(s1), len(s3), len(sj)


def rear_key(grp):
    k = "A" if KEY == "tw" else "upper"
    for r in grp:
        yield r["rear_mm"][k]


def main():
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    import multiprocessing as mp
    t0 = time.time()
    setup()
    m = W["m"]
    print(f"[{TOOL}] 出発点 案{gh.VARIANT}  候補 A・B 各 {n_per}  KP {gh.KP} PRESS {gh.PRESS} MU {gh.MU} impratio {gh.IMPRATIO} "
          f"noslip {gh.NOSLIP} solimp {gh.SOLIMP[:3]} solref {gh.SOLREF} dt {gh.DT}  pair {m.npair} 組  weld "
          f"{int(sum(m.eq_type == mujoco.mjtEq.mjEQ_WELD))} 本  mocap {m.nmocap}  重力 {m.opt.gravity}  "
          f"道具の質量 {[round(float(m.body_subtreemass[m.body(b).id]) * 1000, 1) for b in W['free_bodies']]} g")
    print(f"  A: 角度 ±{A_DEG}度 位置 ±{A_POS * 1000:.0f}mm 向き ±{A_ROT}度  B: 位置 ±{B_POS * 1000:.0f}mm 向き ±{B_ROT}度 "
          f"閉じる速さ {B_SPEED}rad/s 止める力 {B_STOP}N  並列 {WORKERS}", flush=True)
    res, rejected = [], {"A": {}, "B": {}}
    with mp.get_context("spawn").Pool(WORKERS, initializer=init_worker) as pool:
        for method in ("A", "B"):
            got, s0, batch = 0, 0, WORKERS * 8
            while got < n_per:
                for r in pool.map(job, [(method, s) for s in range(s0, s0 + batch)], chunksize=2):
                    if got >= n_per:
                        break
                    if "rejected" in r:
                        rejected[method][r["rejected"]] = rejected[method].get(r["rejected"], 0) + 1
                        continue
                    res.append(r)
                    got += 1
                s0 += batch
                print(f"    {method} {got}/{n_per}  {time.time() - t0:.0f}秒", flush=True)
    # 1バッチずつ並列に回し、seed の小さい順に n_per 個を採る（引き直しは seed の小さい順に数える）
    print(f"  引き直した数 A {rejected['A']}  B {rejected['B']}")
    print("ふるいの段ごと（実測）:")
    tot = [0, 0, 0, 0]
    for method in ("A", "B"):
        c = report(res, method)
        tot = [a + b for a, b in zip(tot, c)]
    surv = [r for r in res if r["passed"] == "arm"]
    order = {"arm": 3, "3s": 2, "1s": 1, "0": 0}

    def score(r):
        m1 = r.get("m1") or r["m0"]
        t_l = r["lost"]["t"] if r["lost"] else TEND
        return (order[r["passed"]], t_l if r["passed"] != "arm" else 0, m1["com_inside"], m1["opposition_deg"], m1["n_contact"])
    if surv:
        surv.sort(key=lambda r: (r["m1"]["com_inside"], r["m1"]["opposition_deg"], r["m1"]["n_contact"]), reverse=True)
        top = surv[:3]
    else:
        top = []
    if len(top) < 3:   # 残りが3つ未満なら、残らなかった候補のうち長く持った順で埋める（画像では「落ちた」と書く）
        top += [r for r in sorted(res, key=score, reverse=True) if r["passed"] != "arm"][:3 - len(top)]
    if surv:
        mains = np.array([r["main"] for r in surv])
        dl = np.array([r["delta_mm"] for r in surv])
        print("  残り方（腕まで残った候補）: 指の角度 rad " + " ".join(
            f"{n.replace('right_', '').replace('_joint', '')} {mains[:, k].mean():.2f}±{mains[:, k].std():.2f}" for k, n in enumerate(gh.MAIN))
              + f"  道具のずらし mm 平均 {np.round(dl.mean(0), 1)} 標準偏差 {np.round(dl.std(0), 1)}  A {sum(r['method'] == 'A' for r in surv)}・B {sum(r['method'] == 'B' for r in surv)}")
    print(f"上位3つ（腕まで残った {len(surv)} 個を 重心の投影・押す向きの角度・接触点の数 で並べ、足りない分は残らなかった候補を長く持った順で埋めた）:")
    for i, r in enumerate(top):
        m1 = r.get("m1") or {}
        print(f"  {i + 1}. {r['method']}#{r['seed']} 段 {r['passed']} 落ち方 {r['lost']} 1秒時点 {m1} 初期 {r['m0']} "
              f"指 {np.round(r['main'], 3).tolist()} ずらし {np.round(r['delta_mm'], 1).tolist()}mm/{r['delta_rot_deg']:.1f}度 後ろ端 {r['rear_mm']}")
    # 画像・動画・JSON
    name = {"tweezers": "ピンセット", "chopsticks": "箸"}[TOOL]
    png = HERE / f"grasp_search_{TOOL}{TAG}.png"
    render_sheet(top, png, f"{name}：物理で探した持ち方 上位3つ（出発点 案{gh.VARIANT}・候補 {len(res)}・溶接なし）")
    vids = []
    for i, r in enumerate(top):
        vp = HERE / f"grasp_search_{TOOL}{TAG}_{i + 1}.mp4"
        again = evaluate(r, video=vp, caption=f"{name} {i + 1}位 {r['method']}#{r['seed']}・溶接なし")
        vids.append(str(vp))
        print(f"  動画 {vp.name}: 撮り直しの結果 段 {again['passed']}（ふるいと同じか: {again['passed'] == r['passed']}）")
    out = HERE / f"grasp_search{TAG}.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    slim = lambda r: {k: v for k, v in r.items() if k not in ("finger_q",)}
    data[TOOL] = dict(
        date="2026-10-21", start_variant=gh.VARIANT, frame="right_base_link（+x 手のひら側・+y 親指側・+z 指の向き）",
        settings=dict(A_DEG=A_DEG, A_POS_MM=A_POS * 1000, A_ROT_DEG=A_ROT, B_POS_MM=B_POS * 1000, B_ROT_DEG=B_ROT, B_SPEED=B_SPEED,
                      B_STOP_N=B_STOP, PRESS=gh.PRESS, KP=gh.KP, MU=gh.MU, dt=gh.DT, seed=SEED, sieve="0-1s 止める / 1-3s 止める / 3-5s 腕10cm 上左下右 / 5-6s 止める・5mm 5度"),
        counts=dict(candidates=tot[0], s1=tot[1], s3=tot[2], arm=tot[3], rejected=rejected),
        main_joints=gh.MAIN, finger_joints=[m.joint(j).name for j in W["finger_jnts"]],
        top=[{**r, "rank": i + 1} for i, r in enumerate(top)],
        all=[slim(r) for r in res], image=png.name, videos=[Path(v).name for v in vids])
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"画像: {png}\n動画: {', '.join(vids)}\nJSON: {out}  （{time.time() - t0:.0f}秒）")
    print(f"SUMMARY {TOOL} 候補 {tot[0]} 1秒残り {tot[1]} 3秒残り {tot[2]} 腕を動かして残り {tot[3]}")


if __name__ == "__main__":
    main()
