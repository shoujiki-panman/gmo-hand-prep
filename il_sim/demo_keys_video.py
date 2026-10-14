"""教材（LeRobot「Imitation Learning in Sim」）の環境で、キーボード操作と同じ動きをプログラムで行い、
「いま押しているキー」の字幕つき動画にする。録画（gym_manipulator）の前に、操作の意味を目で見るための見本。

キーと動きの対応は gym_hil のコード（intervention_utils.py / hil_wrappers.py）から読んだもの:
  ↑=+x ↓=−x ←=+y →=−y  右Shift=上げる 左Shift=下げる  何もしない=1.0
  爪: 右Control が 2.0 を送り、実際は「閉じる」（コード上の名前は open。10/12 実測で箱が持ち上がった）。左Control は 0.0＝開く
  ⚠️ MacBook の内蔵キーボードには右Control が無い

  cd lerobot && .venv/bin/python ../il_sim/demo_keys_video.py   # il_sim/demo_keys.mp4 ができる
"""

import subprocess
from pathlib import Path

import gym_hil  # noqa: F401  環境の登録
import gymnasium as gym
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "demo_keys.mp4"
W, H = 960, 720
FPS = 10                 # 教材の設定と同じ（1秒に10回操作）
REPEAT = 2               # 1回の操作を2コマ見せて、ゆっくりにする
FONT = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", 32)
CLOSE, NOOP = 2.0, 1.0   # 2.0 で爪が閉じる（実測）


def keys_for(move, grip):
    names = []
    for v, plus, minus in zip(move, ["↑", "←", "右Shift"], ["↓", "→", "左Shift"]):
        if v > 0:
            names.append(plus)
        elif v < 0:
            names.append(minus)
    if grip == CLOSE:
        names.append("右Control")
    return "・".join(names) if names else "（何も押さない）"


def main():
    env = gym.make("gym_hil/PandaPickCube-v0")
    env.reset(seed=0)
    base = env.unwrapped
    model, data = base._model, base._data
    tcp = lambda: data.sensor("2f85/pinch_pos").data.copy()      # 爪の間の点
    block = lambda: data.sensor("block_pos").data.copy()         # ピンクの箱

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.45, 0.0, 0.15]
    cam.distance, cam.azimuth, cam.elevation = 1.1, 150, -25

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS * REPEAT // 2), "-i", "-", "-pix_fmt", "yuv420p", str(OUT)],
        stdin=subprocess.PIPE)
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H   # 画面外に描く大きさを広げる
    renderer = mujoco.Renderer(model, H, W)

    def frame(stage, keys):
        renderer.update_scene(data, cam)
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img)
        d.rectangle([0, H - 104, W, H], fill=(30, 30, 35))
        d.text((24, H - 94), stage, font=FONT, fill=(255, 212, 0))
        d.text((24, H - 50), f"押しているキー：{keys}", font=FONT, fill=(230, 230, 230))
        for _ in range(REPEAT):
            ffmpeg.stdin.write(np.asarray(img).tobytes())

    def act(stage, move, grip=NOOP):
        frame(stage, keys_for(move, grip))
        _, _, terminated, truncated, info = env.step(np.array([*move, grip], dtype=np.float32))
        return terminated or truncated, info

    for _ in range(5):
        frame("0  スペースで操作を始める", "スペース（1回だけ）")

    done, info, step = False, {}, 0
    stage = "1  矢印キーで、爪をピンクの箱の真上へ"
    while not done and step < 100:
        d = block() - tcp()
        if stage.startswith("1"):
            move = [np.sign(d[0]) if abs(d[0]) > 0.012 else 0, np.sign(d[1]) if abs(d[1]) > 0.012 else 0, 0]
            if move == [0, 0, 0]:
                stage = "2  左Shift で、爪を箱の高さまで下げる"
                continue
            done, info = act(stage, move)
        elif stage.startswith("2"):
            if d[2] > -0.005:
                stage, closing = "3  右Control で、爪を閉じて箱をつかむ", 0
                continue
            done, info = act(stage, [0, 0, -1])
        elif stage.startswith("3"):
            done, info = act(stage, [0, 0, 0], CLOSE)
            closing += 1
            if closing >= 6:
                stage = "4  右Control を押したまま、右Shift で持ち上げる"
        else:
            done, info = act(stage, [0, 0, 1], CLOSE)
        step += 1

    result = "成功（自動で次の回へ）" if info.get("succeed") else "失敗"
    for _ in range(15):
        frame(f"5  箱が持ち上がった → {result}", "（何も押さない）")
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"保存: {OUT}  結果: {result}  操作回数: {step}")


if __name__ == "__main__":
    main()
