# 5本指ハンドで細い道具を「指の接触だけで」持つ：世の中の作り方（2026-10-21 下調べ）

目的: G1＋BrainCo Revo 2 右手（MuJoCo）で、溶接ではなく指で箸・ピンセットを持たせ、9mm の豆を移す。その作り方を自分で発明しないための下調べ。
表記: 【原文】は論文・公式文書に書いてあること、【実物】は公開リポジトリの XML やコードを自分で開いて確かめた値、【推測】は自分の推測。

## 表1 例ごとの比較

| 例 | 道具の保持 | 操作の方法 | 学習時間 | 成功率 | 出典 |
|---|---|---|---|---|---|
| Learning to Use Chopsticks (SIGGRAPH 2022, MuJoCo 1.5, 30自由度の手) | **指の接触だけ**。箸は free joint で、手から抜け落ちることもある【原文・実物】 | 握りの形は BO と IK で決め、保持と移動は PPO（握りの形からの補正 δq を PD で追従）。学習なしの部分は IK、計画、腕の IK【原文】 | 握りの形の探索 BO は17の持ち方で約6日。移動の学習は1つの握り方あたり約2日（CPU 32コア）【原文】 | 数値の表なし。「最初の数回は必ず成功し、何回か動かすと箸が徐々に抜けていく」原因は MuJoCo の柔らかい接触【原文】 | https://arxiv.org/abs/2205.14313 ／コード https://github.com/chopsticks-research2022/learning2usechopsticks |
| 階層RLでピンセット操作 (arXiv 2507.06822, IROS 2025, MuJoCo, Schunk SVH) | **手に固定**（「tool firmly fixed in the hand」）。実機は親指を差し込む溝つきの3Dプリント製ピンセット。sim でどう固定したか（weld か接触か）は書かれていない【原文】 | 階層RL（SAC）。下位は指の関節位置の差分を 20Hz で出し、上位は腕を動かす【原文】 | 書かれていない | 実機で初回 54.2%、複数回の試行で 70.8%（物体4種×12回）【原文】 | https://arxiv.org/abs/2507.06822 |
| Lessons from Learning to Spin "Pens" (CoRL 2024, Isaac Gym, Allegro) | 指の接触だけ（ペンを回す） | 手で設計した6つの握りの形にノイズを加え、安定したものだけを初期状態に残して PPO。関節目標は PD で追従（制御 20Hz・sim 200Hz）【原文】 | 8192環境で5億ステップ、GPU 1枚で1日未満【原文】 | 実機で 180°回す成功率が 47〜81%【原文】 | https://arxiv.org/abs/2407.18902 |
| UniDexGrasp (CVPR 2023) | 指の接触（持ち上げ） | 握りの形を生成する段と、目標つき RL で実行する段の2段【原文】 | 未確認 | 平均 60% 超【原文】 | https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_UniDexGrasp_Universal_Robotic_Dexterous_Grasping_via_Learning_Diverse_Proposal_Generation_CVPR_2023_paper.pdf |
| MuJoCo Menagerie の手（Shadow / Allegro / LEAP）、Playground の LEAP | 道具を持つデモは見つからなかった。値は接触設定の定番として使う | position アクチュエータ（関節の位置制御） | — | — | https://github.com/google-deepmind/mujoco_menagerie ／ https://github.com/google-deepmind/mujoco_playground |
| BrainCo Revo 2 実機 | ペン・箸・ピンセットを持つ公開映像は見つからなかった。公式PRの例は「マッチを擦る」。指先の素材は公開情報で確認できなかった | 6モーターで11関節。つまみ力 15N 以上、握力 50N 以上（販売店のスペック表）。Touch 版だけ指先に触覚がある | — | — | https://openelab.io/sv/products/brainco-dexterous-hand-revo-2 ／ https://pr.enewspf.com/article/BrainCo-Launches-Revo2-Hand-for-Humanoid-Robotics/68c95ca6b83f4e9fe2c52880 |

## 各例の設定値（公開リポジトリで実物を確認）

**箸論文** `data/xml_templates/standard/0/sim.xml` と `envs/hand_env.py`【実物】
- `<option timestep="0.01" iterations="20" impratio="1" noslip_iterations="100">`。別の XML（chopsticks.xml）は `impratio="10" noslip_iterations="20"`
- 箸は capsule で半径 4mm・長さ 26cm、free joint、`condim="4"`、`friction="1 0.005 0.0001"`、`solimp="0.99 0.99 0.01"`、`solref="0.01 1"`
- 指先と箸の接触は `<pair ... condim="4" margin="0" gap="0">` で組ごとに明示している。手の既定値は `friction="1 0.005 0.001" condim="3"`、関節 `damping=0.05 armature=0.001`
- 制御は自前の PD トルク。指の関節は **kp=1, kd=0.1, トルク上限 100**、腕は kp=5000, kd=500
- 論文本文【原文】: solimp を (0.85,0.9,0.001) にすると「めり込みが大きくて失敗」。impratio は1より大きいと摩擦が「硬く」なり、中くらいの値が一番よかった

**Menagerie / Playground の手**【実物】
- Shadow: `cone="elliptic" impratio="10"`、position の kp=10、forcerange ±10（一部 ±1）
- Allegro: `cone="elliptic" impratio="10"`、position の kp=1、joint damping 0.1
- LEAP（Menagerie）: `integrator="implicitfast" cone="elliptic" impratio="100"`、geom の既定値 `friction=".2" solimp="0.999 0.999 0.001 0.0001 1" solref="0.0001 1"`、position の kp=3, kv=0.01
- LEAP（Playground、MJX 学習用）: `timestep="0.01"`、`friction="0.2" solref="0.02 1.5" solimp="0.9 0.99 0.001"`

**BrainCo 公式 Revo2 の MJCF** `BrainCoTech/brainco-description/revo2_system/mjcf/revo2_right.xml`【実物】
- `<option cone="elliptic" impratio="10"/>` だけで、**actuator も、関節を連動させる equality も、摩擦・solref の既定値も入っていない**
- 11関節の `actuatorfrcrange` は親指 0.5〜1.1、他の指 2（N·m）
- Unitree 版 URDF は `unitreerobotics/unitree_ros/robots/g1_with_brainco_hand/` にある。依頼元が使っている Unitree 公式モデルとの差は確認していない

**MuJoCo 公式「Preventing slip」** https://mujoco.readthedocs.io/en/stable/modeling.html 【原文】
- 遅い滑りは柔らかい接触モデルでは避けられず、完全には消せない
- 対策1: 楕円コーン（elliptic）＋大きな impratio
- 対策2: `noslip_iterations` を正の値にする。「1〜3で通常は十分」。ただし不安定になることがある
- 力が摩擦コーンの外に出るなら、法線力・摩擦係数を上げるか、condim を 4 か 6 にする
- 振動するなら timestep を下げるか armature を足し、implicit 系の積分器を使う
- solref の時定数は timestep の2倍以上にする

## 定番手順（複数の例に共通するもの）

1. **初期配置は「形を決めてから置く」。** 握りの形（指の角度）と道具の位置を先に決めてから sim を始める。箸論文は指先の接触点から IK で解き、めり込みには罰則をかけた。ペン論文は手で作った形にノイズを加え、安定したものだけを残した。どちらも、手を閉じていって握らせる作り方ではなかった【原文】
2. **接触は楕円コーン＋impratio を上げる。** Menagerie の手はすべて elliptic で、impratio は 10〜100。指先と道具の組は `<pair>` で condim 4 にし、friction は 1 前後（LEAP は 0.2）。それでも滑るなら noslip を 1〜3【実物・原文】
3. **指は位置制御で、関節目標を接触より少し内側に置く。** 握る力は kp×誤差で決まり、forcerange で頭打ちになる。指の kp は小さい（1〜10）【実物】。目標を内側にずらして押しつける使い方は、箸論文の「握りの形＋補正 δq」と構造が同じ【推測】
4. **学習は操作の段だけに入れる。** 箸論文は握りの形の評価と移動に PPO、ピンセット論文は道具を固定して操作だけを学習、UniDexGrasp は握りの形の生成と実行を分けている【原文】
5. **落ちたときの切り分け。** めり込みが大きいなら solimp を硬くする（0.9 以上）。ゆっくり滑るなら impratio → noslip → 摩擦の順に試す。振動するなら timestep を下げるか armature を足す。何回か動かすと抜けるのは箸論文でも起きた既知の現象で、論文は計画し直しと持ち直しで対処した【原文】

## 「箸・ピンセットを指だけで保持」はどれくらい難しいか

- **実測（論文の記述）:** 指だけで箸を持った唯一の例（箸論文）は、PPO で学習しても「何回か動かすと徐々に抜ける」と書いている。ピンセット論文は「保持が肝心なので固定した」と書き、指だけの保持を避けた
- **見つからなかったもの:** MuJoCo で5本指が細い棒を学習なし・指の接触だけで持ち続ける公開例は見つからなかった（1時間の範囲で）。存在しないという証明ではない
- **推測:** 保持するだけ（腕を止めて数秒）なら、学習なしで成り立つ見込みがある。箸論文でも最初の数回は必ず成功しているから。難しいのは腕を動かしたときと、豆をつまむときの保持。Revo2 はモーター6つで遠位関節が連動するため、人の箸の持ち方（親指・人差し指・中指を別々に動かす）は再現しにくいはず。2本の箸よりも、1つの物体として扱えるピンセット（ばね付きのヒンジ）のほうが楽そうだ

## 最初の1段（握って保つ・学習なし）の具体案

1. option は `cone="elliptic" impratio="10"`（BrainCo の MJCF のまま）、`integrator="implicitfast"`。滑ったら `noslip_iterations="2"` を足す（MuJoCo 公式文書）
2. Revo2 には actuator が無いので、6モーターぶんの `<position>` と、遠位を近位に連動させる `<equality><joint polycoef>` を足す。forcerange は MJCF の actuatorfrcrange に合わせる（BrainCo 公式 MJCF の実物）
3. 道具は free joint の capsule（箸なら半径 4mm）。指先と道具の組だけ `<pair condim="4" friction="1 0.005 0.0001">` にする（箸論文の実物）
4. 初期状態は keyframe にする。指の角度と道具の位置を先に決め、めり込みが小さい状態から始める。指の目標角は接触より少し内側に置く（箸論文・ペン論文の「形を先に決める」手順）
5. 合否は、腕を止めて5秒間、重力下で道具がどれだけずれるか（mm）と接触の数を測って決める。落ちたら「定番手順 5」の順で直す。腕を動かす段は保持が通ってから
6. ピンセットを先にやり、箸は後にする（【推測】上の理由による）

## 出典
- https://arxiv.org/abs/2205.14313 / https://github.com/chopsticks-research2022/learning2usechopsticks
- https://arxiv.org/abs/2507.06822
- https://arxiv.org/abs/2407.18902
- https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_UniDexGrasp_Universal_Robotic_Dexterous_Grasping_via_Learning_Diverse_Proposal_Generation_CVPR_2023_paper.pdf
- https://mujoco.readthedocs.io/en/stable/modeling.html （Preventing slip）
- https://github.com/google-deepmind/mujoco_menagerie （leap_hand / shadow_hand / wonik_allegro の right_hand.xml）
- https://github.com/google-deepmind/mujoco_playground （leap_hand/xmls/leap_rh_mjx.xml）
- https://github.com/BrainCoTech/brainco-description （revo2_system/mjcf/revo2_right.xml）
- https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_with_brainco_hand
- https://openelab.io/sv/products/brainco-dexterous-hand-revo-2 ／ https://pr.enewspf.com/article/BrainCo-Launches-Revo2-Hand-for-Humanoid-Robotics/68c95ca6b83f4e9fe2c52880
