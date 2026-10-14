"""学習した ACT にキューブをつかませ、成功率を出す（教材 Imitation Learning in Sim の評価の代わり）。

教材の評価プログラム lerobot/rl/eval_policy.py は LeRobot 0.6.2 では設定の項目名が合わず動かない（10/12 実測）。
中身は同じ「リセット → AI が動きを選ぶ → 1歩進める」を N 回くり返して成功率を出す。
観測と動きの変換は、録画（gym_manipulator）と同じ部品を使う。LeRobot 本体は書き換えない。
画面は出さず、最初の数回を動画（eval_act.mp4）に残す。

  cd <LeRobot を clone した場所> && .venv/bin/python <このファイル> <学習結果 pretrained_model のパス> [回数] [何歩ごとに見直すか]

何歩ごとに見直すか（ACT の n_action_steps）: 学習時の設定は100＝最初に100歩ぶんを決めて見直さない。
1 にすると毎歩カメラを見て決め直す（学習のやり直しは不要）。10/14 実測で 100 だと「何もしない」を出し続けた。
"""

import subprocess
import sys
from pathlib import Path

import draccus
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import TransitionKey
from lerobot.rl.gym_manipulator import (
    GymManipulatorConfig,
    make_processors,
    make_robot_env,
    reset_and_build_transition,
    step_env_and_process_transition,
)

HERE = Path(__file__).parent
CONFIG = HERE / "env_config.json"   # 録画で使った設定（環境の作り方を同じにする）
OUT = HERE / "eval_act.mp4"  # main で見直し間隔に応じて名前を変える
W, H, FPS = 640, 480, 10
VIDEO_EPISODES = 3
FONT = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", 26)


def main():
    policy_path = Path(sys.argv[1]).expanduser()
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10   # 教材の eval_policy と同じ10回
    n_action_steps = int(sys.argv[3]) if len(sys.argv) > 3 else None

    cfg = draccus.parse(GymManipulatorConfig, config_path=str(CONFIG), args=[])
    cfg.env.task = "PandaPickCube-v0"   # 画面とキーボードの無い版。中身（ロボット・箱・カメラ）は同じ
    device = "mps"

    global OUT
    if n_action_steps:
        OUT = HERE / f"eval_act_{policy_path.parent.name}_n{n_action_steps}.mp4"
    env, _ = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, None, cfg.env, device)

    policy = ACTPolicy.from_pretrained(policy_path)
    policy.to(device).eval()
    if n_action_steps:
        policy.config.n_action_steps = n_action_steps
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    base = env.unwrapped
    base._model.vis.global_.offwidth, base._model.vis.global_.offheight = W, H
    renderer = mujoco.Renderer(base._model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.45, 0.0, 0.15]
    cam.distance, cam.azimuth, cam.elevation = 1.1, 150, -25
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", str(OUT)], stdin=subprocess.PIPE)

    def frame(text):
        renderer.update_scene(base._data, cam)
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img)
        d.rectangle([0, H - 48, W, H], fill=(30, 30, 35))
        d.text((16, H - 40), text, font=FONT, fill=(255, 212, 0))
        ffmpeg.stdin.write(np.asarray(img).tobytes())

    results = []
    for ep in range(n_episodes):
        transition = reset_and_build_transition(env, env_processor, action_processor)
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        success, steps = False, 0
        while True:
            obs = {k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k.startswith("observation")}
            with torch.no_grad():
                action = postprocessor(policy.select_action(preprocessor(obs)))
            transition = step_env_and_process_transition(
                env, transition, action, env_processor, action_processor)
            steps += 1
            info = transition[TransitionKey.INFO]
            success = success or bool(info.get("succeed", False))
            if ep < VIDEO_EPISODES:
                frame(f"第{ep + 1}回  {steps}歩目" + ("  → 成功" if success else ""))
            if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
                break
        results.append(success)
        print(f"第{ep + 1}回: {'成功' if success else '失敗'}（{steps}歩）", flush=True)

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"成功率: {sum(results)}/{n_episodes} = {sum(results) / n_episodes:.0%}")
    print(f"動画: {OUT}（最初の{min(VIDEO_EPISODES, n_episodes)}回）")


if __name__ == "__main__":
    main()
