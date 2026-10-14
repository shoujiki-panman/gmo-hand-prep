# GMOロボットハンドコンテスト 事前学習の記録

[第1回 GMOロボットハンドコンテスト](https://ai-robotics.gmo/lp/gmo-robot-hand-competition/)（5本指ハンドで炒り豆を5分間移す）に向けて、応募前の数日間に行った事前学習の記録です。

- 北京の世界大会にある同じ種目を調べました。
- 大会の機体 G1-D に近い「Unitree G1 ＋ BrainCo の5本指ハンド」をシミュレーターで動かしました。
- [LeRobot](https://github.com/huggingface/lerobot) の公式教材で「お手本 → 学習 → 評価」を一度回しました。

すべて MacBook Air（M5・32GB）上の MuJoCo シミュレーションで行い、実機は使っていません。

## 結果のまとめ

| 学習の段階 | 成功 | 様子 |
| --- | --- | --- |
| 2万歩 | 0 / 10 | まだつかめない |
| 4万歩 | **10 / 10** | 40歩でつかんで持ち上げる |
| 6万歩 | **10 / 10** | 成功 |
| 8万歩 | **10 / 10** | 成功（80歩） |
| 10万歩 | 0 / 10 | つかむが持ち上げない |

| 4万歩：下げる → つかむ → 持ち上げる | 10万歩：つかんだまま止まる |
| --- | --- |
| ![4万歩の評価](media/act_40k.gif) | ![10万歩の評価](media/act_100k.gif) |

最後まで学習した結果がいちばん良いとは限りませんでした。途中の段階も評価して選ぶ必要があります。

## 1. 北京で同じ種目がすでに行われていた

2026年8月の World Humanoid Robot Games（北京）に、5本指ハンドが「ピンセットで大豆をつまむ」種目（镊子夹豆 Tweezers Bean Picking）がありました。

| 項目 | 北京（WHRG 2026） | GMO 第1回 |
| --- | --- | --- |
| 課題 | ピンセットで大豆をつかむ・運ぶ・離す | 炒り豆を1粒ずつ皿から皿へ |
| 豆 | 直径 5mm の大豆 | 炒り豆（大きさの記載なし） |
| 制限時間 | 5分 | 5分 |
| 自律と遠隔操作 | 自律 ×1、遠隔操作 ×0.5 | 記載なし |
| 上位の結果 | 1位 全自律（OmniHand）、3位 遠隔操作で 71粒 | — |

優勝チームの粒数と具体的な手法は公開されていませんでした。

出典：[WHRG 2026 公式の種目一覧（PDF）](https://www.whrgoc.com/resources//uploads/20260822/1787382689139022892.pdf) ／ [Gasgoo](https://autonews.gasgoo.com/articles/news/opening-day-of-dexterous-hand-special-event-at-world-humanoid-robot-games-critical-point-wins-three-golds-2092584395855978497) ／ [中国経営報](https://news.qq.com/rain/a/20260826A0DKBB00) ／ [3位チームの記事](https://news.qq.com/rain/a/20260826A0BTNO00) ／ [新浪（制限時間）](https://news.sina.com.cn/c/2026-08-25/doc-inipnpyw9744459.shtml)

## 2. G1 ＋ BrainCo の5本指をシミュレーターで動かす

大会サイトに載っている機体は G1-D です。[GMO ヒューマノイド.shop](https://gmo-humanoid.shop/unitree/g1d/) で G1-D に付けられる5本指は BrainCo Revo 2 だけでした。そこで Unitree が公開している「G1 ＋ BrainCo の手」のモデルを MuJoCo で表示し、右手の6つの指令を一つずつ動かしました。

![G1 と右手の6つの指令](media/g1_demo.gif)

| 指令 | 動き |
| --- | --- |
| thumb_metacarpal | 親指が手のひらの方へ倒れる |
| thumb_proximal | 親指が曲がる |
| index / middle / ring / pinky _proximal | それぞれの指が曲がる |

指先の関節は根元に連動するので、関節の数（11）より指令の数（6）が少なくなっています。動画は関節の角度を直接入れて形を動かしたもので、物理の計算はしていません。ラボの実機に Revo 2 が付いているかは確認していません。

## 3. お手本から学ぶ流れを1周する

LeRobot の公式教材 [Imitation Learning in Sim](https://huggingface.co/docs/lerobot/main/en/il_sim) を使いました。ロボットアームがキューブをつかんで持ち上げる課題です。

1. **お手本**：同じ教材で公開されているお手本 [Kim-Ziho/il_gym_day1_v2](https://huggingface.co/datasets/Kim-Ziho/il_gym_day1_v2)（50回、すべて成功）を使いました。自分でキーボード操作して録る手順は、Mac ではアームが動かず断念しました。
2. **学習**：ACT という方式で、教材どおり 10万歩学習しました。Mac（M5・MPS）で1歩 約0.35秒、10万歩で約10時間でした。
3. **評価**：教材の評価プログラムは今の LeRobot（0.6.2）では動かなかったため、同じ内容（10回やらせて成功数を数える）を [il_sim/eval_act.py](il_sim/eval_act.py) で組み直しました。2万歩ごとに保存した段階をそれぞれ評価しています。

キー操作の意味を確かめるために作った見本（プログラムでキー操作と同じ動きをさせたもの）：

![キー操作の見本](media/demo_keys.gif)

### 分かったこと

- 最後まで学習した結果（10万歩）は失敗し、途中の 4万〜8万歩は成功しました。途中の段階も評価する必要があります。
- 教材どおり「最初に100歩分の動きをまとめて決める」設定では、AI はずっと「何もしない」を出し続けました。「毎歩カメラを見て決め直す」設定（`n_action_steps=1`）に変えると、下げる・つかむまで進みました。上の表はこの設定での結果です。
- キューブの位置は毎回同じなので、10回の評価は実質同じ動きのくり返しです。位置を変えた評価はまだです。
- お手本はキーボード操作のため、ほとんどのコマが「何もしない」で、操作は一瞬だけでした。これが学習を難しくしていると考えています（推測）。

### 教材どおりでは動かなかったところ（LeRobot 0.6.2、2026年10月時点）

- 教材の設定見本 `env_config.json` の `gripper_penalty_in_reward` は今の版では使えず、外さないと起動しません。
- 爪を閉じるキーは右 Control でした。コード上の名前は open（開く）ですが、実際には爪が閉じました。MacBook の内蔵キーボードには右 Control がありません。
- `~/Documents` の下に置くと、仮想環境の `.pth` ファイルに Mac の隠しファイル印が付いて `No module named 'lerobot'` になりました。iCloud 同期の影響とみています（推測）。別の場所に置くと直りました。

## 次にやること（開発期間に入ったら）

- まず遠隔操作で豆を運べるようにする。点数の保険になり、その操作の記録がそのまま自律の AI のお手本になります。
- 同じお手本で自律の AI を学習させ、5分で何粒運べるかを遠隔操作と並べて比べ、本番の方式を数字で決めます。
- ピンセットの持ち方とカメラの位置を早めに固めます（北京の3位チームは持ち方を変えて 33粒 → 71粒）。
- 実機での流れは Unitree 公式の [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate)（遠隔操作で録る）と [unitree_lerobot](https://github.com/unitreerobotics/unitree_lerobot)（学習）を想定しています。

## 再現手順

LeRobot（0.6.2）を Python 3.12 で入れます（`~/Documents` の外に置く）。

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv venv --python 3.12
uv pip install -e ".[hilserl,training]"
```

学習（教材どおり。Mac では `--policy.device=mps`）：

```bash
.venv/bin/lerobot-train --dataset.repo_id=Kim-Ziho/il_gym_day1_v2 --policy.type=act \
  --output_dir=outputs/train/il_sim_test --job_name=il_sim_test \
  --policy.device=mps --policy.push_to_hub=false --wandb.enable=false
```

評価（学習結果のパス・回数・何歩ごとに見直すか）：

```bash
.venv/bin/python <このリポジトリ>/il_sim/eval_act.py outputs/train/il_sim_test/checkpoints/040000/pretrained_model 10 1
```

G1 の動画は、Unitree の [unitree_ros](https://github.com/unitreerobotics/unitree_ros) から `robots/g1_with_brainco_hand` を `sim/unitree_ros/` に取ってきてから、`python sim/g1_demo_video.py` で作れます（mujoco 3.15 で確認）。

## ファイル

| パス | 中身 |
| --- | --- |
| `il_sim/env_config.json` | 教材の設定見本を今の版に合わせたもの |
| `il_sim/eval_act.py` | 評価（10回・成功数・動画） |
| `il_sim/demo_keys_video.py` | キー操作の見本動画を作る |
| `il_sim/results/` | 評価の記録 |
| `sim/g1_demo_video.py` | G1 ＋ BrainCo の右手を動かす動画を作る |
| `media/` | README の動画（GIF） |

## 使ったもの

- [LeRobot](https://github.com/huggingface/lerobot)（Apache-2.0）と gym-hil
- お手本 [Kim-Ziho/il_gym_day1_v2](https://huggingface.co/datasets/Kim-Ziho/il_gym_day1_v2)（Apache-2.0）
- [unitree_ros](https://github.com/unitreerobotics/unitree_ros) の G1 ＋ BrainCo モデル（BSD-3-Clause）
- [MuJoCo](https://github.com/google-deepmind/mujoco)

このリポジトリのコードは MIT ライセンスです。
