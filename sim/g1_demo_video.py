"""G1（BrainCo の手つき・Unitree 公式モデル）が机の前に立つ場面で、右手の6つの指令を1つずつ見せる動画。

形だけを動かす（関節の角度を直接入れる）。物理計算はしていない。
G1-D は G1 の上半身を台座に載せた機体。脚はこのモデルに残っているが、大会の機体には無い。

  .venv/bin/python g1_demo_video.py   # g1_demo.mp4 ができる
"""

import math
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
G1 = HERE / "unitree_ros" / "robots" / "g1_with_brainco_hand"
URDF = G1 / "g1_29dof_mode_15_brainco_hand.urdf"
OUT = HERE / "g1_demo.mp4"
FPS = 30
W, H = 960, 720
FONT = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", 32)

TABLE_TOP = 0.72
PLATES = [(0.45, -0.20), (0.45, 0.0)]   # 右手の前に豆の皿、その左に移す先の皿

# 右手の6つの指令（URDF で mimic されていない関節）と字幕
STEPS = [
    ("right_thumb_metacarpal_joint", "親指の付け根を回す（手のひらの方へ倒れる）"),
    ("right_thumb_proximal_joint", "親指を曲げる"),
    ("right_index_proximal_joint", "人差し指を曲げる"),
    ("right_middle_proximal_joint", "中指を曲げる"),
    ("right_ring_proximal_joint", "薬指を曲げる"),
    ("right_pinky_proximal_joint", "小指を曲げる"),
]
# 指先の関節は根元に連動する（URDF の mimic の倍率）
MIMIC = {
    "right_thumb_proximal_joint": ("right_thumb_distal_joint", 1.0),
    "right_index_proximal_joint": ("right_index_distal_joint", 1.155),
    "right_middle_proximal_joint": ("right_middle_distal_joint", 1.155),
    "right_ring_proximal_joint": ("right_ring_distal_joint", 1.155),
    "right_pinky_proximal_joint": ("right_pinky_distal_joint", 1.155),
}
ARM_POSE = {  # 両腕を机の上へ出す
    "right_shoulder_pitch_joint": -0.5, "right_elbow_joint": 0.6,
    "left_shoulder_pitch_joint": -0.5, "left_elbow_joint": 0.6,
}


def build():
    text = URDF.read_text().replace('meshdir="meshes"', f'meshdir="{G1.resolve()}"')
    robot = mujoco.MjSpec.from_string(text)

    spec = mujoco.MjSpec()
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.97, 0.97, 0.98], rgb2=[0.85, 0.87, 0.9], width=256, height=256)
    w = spec.worldbody
    w.add_light(pos=[1, -1, 2.5], dir=[-1, 1, -2.5], castshadow=True)
    w.add_light(pos=[-1, 1, 2.5], dir=[1, -1, -2.5], castshadow=False, diffuse=[0.4, 0.4, 0.4])
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.01], rgba=[0.88, 0.88, 0.9, 1])
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.4, 0.02],
               pos=[0.55, 0, TABLE_TOP - 0.02], rgba=[0.6, 0.5, 0.4, 1])
    rng = np.random.default_rng(0)
    for k, (x, y) in enumerate(PLATES):
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.07, 0.004, 0],
                   pos=[x, y, TABLE_TOP + 0.004], rgba=[0.97, 0.97, 0.97, 1])
        if k == 0:   # 豆（直径 約8mm の仮の大きさ）
            for _ in range(15):
                r, a = 0.05 * math.sqrt(rng.random()), 2 * math.pi * rng.random()
                w.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.004, 0, 0],
                           pos=[x + r * math.cos(a), y + r * math.sin(a), TABLE_TOP + 0.012],
                           rgba=[0.75, 0.55, 0.3, 1])
    # ピンセット（形だけ）
    w.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.004, 0.002],
               pos=[0.35, -0.33, TABLE_TOP + 0.002], rgba=[0.7, 0.7, 0.75, 1])

    spec.attach(robot, frame=w.add_frame(pos=[0, 0, 0.80]), prefix="")
    spec.visual.global_.offwidth = W
    spec.visual.global_.offheight = H
    model = spec.compile()
    return model, mujoco.MjData(model)


def setj(model, data, name, value):
    data.qpos[model.joint(name).qposadr[0]] = value
    if name in MIMIC:
        child, mult = MIMIC[name]
        data.qpos[model.joint(child).qposadr[0]] = mult * value


def caption(frame, line1, line2):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, H - 104, W, H], fill=(30, 30, 35))
    d.text((24, H - 94), line1, font=FONT, fill=(255, 212, 0))
    d.text((24, H - 50), line2, font=FONT, fill=(230, 230, 230))
    return np.asarray(img)


def main():
    model, data = build()
    for name, v in ARM_POSE.items():
        setj(model, data, name, v)
    mujoco.mj_forward(model, data)

    wide = mujoco.MjvCamera()
    wide.lookat[:] = [0.25, -0.05, 0.85]
    wide.distance, wide.azimuth, wide.elevation = 2.0, 150, -22
    close = mujoco.MjvCamera()   # 親指の回転が見える角度（右手を外側から）
    close.lookat[:] = data.xpos[model.body("right_base_link").id]
    close.distance, close.azimuth, close.elevation = 0.35, 270, -30

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", str(OUT)],
        stdin=subprocess.PIPE)

    with mujoco.Renderer(model, H, W) as r:
        # 1. 全体
        for _ in range(int(3.5 * FPS)):
            r.update_scene(data, wide)
            ffmpeg.stdin.write(caption(
                r.render(), "G1 の上半身＋BrainCo の5本指（大会の G1-D に近い構成）",
                "机の上に豆の皿と移す先の皿・ピンセット（形だけ）").tobytes())
        # 2. 右手の6つの指令
        for i, (name, text) in enumerate(STEPS):
            top = model.jnt_range[model.joint(name).id, 1]
            n = int(3.0 * FPS)
            for k in range(n):
                v = top * math.sin(math.pi * k / (n - 1))
                setj(model, data, name, v)
                mujoco.mj_forward(model, data)
                r.update_scene(data, close)
                ffmpeg.stdin.write(caption(
                    r.render(), f"{i + 1}/6  {text}",
                    f"指令 {name.removeprefix('right_').removesuffix('_joint')} = {v:.2f}"
                    "（形だけ・物理なし）").tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"保存: {OUT}")


if __name__ == "__main__":
    main()
