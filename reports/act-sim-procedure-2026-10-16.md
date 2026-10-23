# 録る→学習→測る 手順書（シミュ版・試験10で1周した実物）

2026-10-16。MuJoCo＋LeRobot（`~/Developer/gmo-hand/lerobot` の `.venv`、Python 3.12）。12月に実機で同じ1周を回すための下書き。
数字はシミュのもので、本番の見込みではない。

## 0. 前提

- 作業場所: すべて `cd ~/Developer/gmo-hand/lerobot` から。`<work>` は `~/Documents/Codex/2026-06-24/handoff-next-chat-2026-06-24/work/gmo-hand-learning`
- 学習と評価は `caffeinate -i` を頭に付ける（Mac が眠ると止まる）
- 名前は毎回新しくする（`tanuma/chop_rand9` のように）。既存のお手本・学習結果を上書きしない

## 1. 録る（実測 12.2分・50本）

```
.venv/bin/python <work>/chop_collect_rand.py 50
```

- 1回ごとに条件を乱数で決め（摩擦・直径・質量・位置・3割で途中のずらし）、追従制御で動かし、**移せた回だけ**保存する
- 出るもの: `~/.cache/huggingface/lerobot/tanuma/chop_rand9`（20Hz・1本160コマ）と、1本ごとの条件の表 `<work>/chop_rand9_episodes.csv`
- 中身: 観測＝腕7関節の角度＋箸の角度＋画像2台（128px）、指示＝次の腕7関節＋箸の角度
- ハマりどころ: 動画の書き出しを並列にすると子プロセスが落ちる → `save_episode(parallel_encoding=False)`
- 確かめること: CSV で条件が本当にばらついているか（最小〜最大）・何本が成功したか

**実機で置き換わる部分**
- お手本の録り方: プログラムの追従制御 → **VR 遠隔操作で人が動かす**。1本ごとの条件（豆の種類・置き場所・途中で動いたか）は人がメモするか、録画から後で付ける
- 観測の中身: シミュの角度 → 実機の関節センサー／画像 → 実機のカメラ（手首カメラがあるなら、豆が何ピクセルに写るかを先に測る）
- 成功だけ残す判断: シミュは自動判定 → 実機は録った人が見て決める

## 2. 学習（実測 61.4分・1万回）

```
caffeinate -i .venv/bin/lerobot-train --dataset.repo_id=tanuma/chop_rand9 --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=16 --policy.device=mps --policy.push_to_hub=false --steps=10000 --save_freq=2500 --batch_size=8 --num_workers=4 --log_freq=500 --seed=1000 --output_dir=../outputs/train/chop_rand9 --job_name=chop_rand9 --wandb.enable=false > ../outputs/train_chop_rand9.log 2>&1
```

- 2,500回ごとに保存 → `../outputs/train/chop_rand9/checkpoints/{002500,005000,007500,010000}`
- 平均 0.37秒/回（Apple M5・mps）。途中で速さが半分ほどに落ちる時間帯があった
- 確かめること: ログの loss が下がっているか（今回 4.33 → 0.16）・保存4つがあるか
- 実機でも同じコマンド。変わるのはお手本の名前と、観測の数（関節の数・カメラの台数）だけ

## 3. 測る（実測 保存ごと10回で3〜4分）

```
caffeinate -i .venv/bin/python <work>/chop_eval_rand.py ../outputs/train/chop_rand9/checkpoints/010000/pretrained_model 10 16
caffeinate -i .venv/bin/python <work>/chop_eval_rand.py ../outputs/train/chop_rand9/checkpoints/010000/pretrained_model 10 16 1.0,0.015,0   # 1.0秒に豆を x+1.5cm 動かす
```

- 評価の条件はお手本と同じ幅、乱数の seed だけ別（お手本 0・評価 12345）。同じ seed で測ると覚えた場面を測ってしまう
- 16手ずつ実行して見直す（ACT の出力は 50手先まで、そのうち 16手だけ使う）
- 1回目だけ動画 `<work>/chop_eval_rand_<保存>_1.mp4`。失敗はどの段で落ちたか（はさむまで・運ぶ・離す）を1行ずつ出す
- 確かめること: 動画で 3.5秒（持ち上げ）と 7.4秒（皿2の上）を見る。数字だけで成功と書かない

**実機で置き換わる部分**
- AI の指示をシミュの腕に入れる → 実機の腕に送る（安全のため最初は速度を落とす・止めるボタンを手元に）
- 成功の判定と途中の乱れ: 自動 → 人が見て記録する。乱れは人が豆を指でずらす

## 4. 今回の結果（試験10・実測）

| 保存 | 移せた | 持ち上げ |
| --- | --- | --- |
| 2,500 | 0/10 | 0/10 |
| 5,000 | 3/10 | 3/10 |
| 7,500 | 1/10 | 1/10 |
| 10,000 | 3/10 | 5/10 |
| 10,000＋乱れ | 0/10 | 0/10 |

失敗はほぼ「はさむ」まで。お手本（追従制御）は同じばらつきで 50/50。詳しくは README の「ばらつき入りの見本で AI（ACT）を1周」。

## 5. 所要時間のまとめ（実測）

| 段 | 時間 |
| --- | --- |
| 録る（50本） | 12分 |
| 学習（1万回） | 61分 |
| 測る（保存4つ×10回＋乱れ10回） | 16分 |
| 合計 | 約1時間35分 |

実機ではお手本を人が録るので、録る段がいちばん長くなる見込み（推測。1本あたりの時間は実機で測る）。
