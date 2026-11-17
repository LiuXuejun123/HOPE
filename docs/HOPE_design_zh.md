# HOPE 代码详细设计文档（面向强化学习入门者）

> 对应论文：*HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios*（arXiv:2405.20579）
> 本文档基于仓库 `src/` 下的源码整理。所有函数都已补充中文 docstring，在 VSCode 里鼠标悬停即可查看。

---

## 0. 怎么读这份文档

建议按下面顺序阅读，每读一节就打开对应源码对照：

| 顺序 | 章节 | 对应源码 |
|---|---|---|
| 1 | §1 项目要解决什么问题 | `readme.md` |
| 2 | §2 强化学习概念速查（结合本项目） | — |
| 3 | §3 整体架构与一次 step 的数据流 | 全局 |
| 4 | §4 环境：车、地图、观测、动作、奖励 | `src/env/` |
| 5 | §5 动作掩码 | `src/model/action_mask.py` |
| 6 | §6 网络结构 | `src/model/network.py`, `attention.py` |
| 7 | §7 PPO 算法 | `src/model/agent/ppo_agent.py` |
| 8 | §8 混合策略（RL + Reeds-Shepp） | `src/model/agent/parking_agent.py`, `env/reeds_shepp.py` |
| 9 | §9 训练流程 | `src/train/train_HOPE_ppo.py` |
| 10 | §10 SAC 版本的差异 | `sac_agent.py`, `train_HOPE_sac.py` |
| 11 | §11 评估 | `src/evaluation/` |
| 12 | §12 超参数一览 | `src/configs.py` |
| 13 | §13 阅读时需要注意的地方（含疑似 bug） | — |

**可以先当黑盒的部分**：`reeds_shepp.py`（经典几何算法）、`lidar_simulator.py` 和 `action_mask.py` 里的向量化求交代码、`parking_map_normal.py` 的随机场景生成细节、`network.py` 中自编码器相关类。

---

## 1. 项目要解决什么问题

**任务**：给定一个停车场景（起点位姿、目标车位、障碍物），规划一条让车从起点开进车位、全程不碰撞的路径。车可以前进，也可以倒车，转向角有上限（不能原地转圈）。

**HOPE 的三个核心设计**：

1. **混合策略（Hybrid Policy）**：远离车位时由强化学习（RL）智能体一步步决策；一旦离车位足够近（< 10 m），并且能找到一条无碰撞的 **Reeds-Shepp 曲线**（一种解析的最短路径）直达车位，就切换为直接执行这条曲线。RL 负责"复杂的探索"，几何曲线负责"精确的最后一段"。
2. **动作掩码（Action Mask）**：根据激光雷达读数，算出每个候选动作"走多远会撞"，只在安全动作里采样。这让 RL 探索时少走很多"撞墙"的弯路。
3. **Transformer 融合多模态观测**：激光雷达、目标位置、图像、动作掩码各自编码成一个向量（token），用自注意力融合后再输出动作。

---

## 2. 强化学习概念速查（结合本项目）

| 概念 | 一句话解释 | 在本项目中是什么 |
|---|---|---|
| 环境 Environment | 智能体交互的"世界" | `CarParking`（`env/car_parking_base.py`）+ 包装器 `CarParkingWrapper` |
| 智能体 Agent | 做决策并学习的一方 | `PPOAgent` / `SACAgent`，外面再包一层 `ParkingAgent` |
| 观测 Observation / 状态 State | 智能体每一步能看到的信息 | dict：`lidar`、`target`、`img`、`action_mask` |
| 动作 Action | 智能体每一步的输出 | `[转角, 速度]`，网络输出在 `[-1,1]`，再映射到真实范围 |
| 奖励 Reward | 环境对这一步好坏的打分 | 靠近车位、与车位重叠变多则奖励；到达 +50；超时 -1（再 ×0.1） |
| Episode | 从 reset 到结束的一整次尝试 | 一次停车过程，最多 200 步 |
| 策略 Policy π(a\|s) | 观测到动作的映射（通常是一个概率分布） | actor 网络输出高斯分布均值，`log_std` 为独立可学习参数 |
| 价值函数 V(s) | 从状态 s 开始，未来累计奖励的期望 | critic 网络 |
| 折扣因子 γ | 未来奖励打多少折 | `GAMMA = 0.98` |
| 优势 Advantage A(s,a) | 这个动作比"平均水平"好多少 | 用 GAE 估计（§7） |
| On-policy / Off-policy | 是否只能用当前策略采集的数据来训练 | PPO 是 on-policy（数据用完即丢）；SAC 是 off-policy（经验回放池重复使用） |
| Actor-Critic | actor 负责选动作，critic 负责打分指导 actor | PPO 和 SAC 都属于这一类 |

一个最小的 RL 交互循环就是：

```python
obs = env.reset()
while not done:
    action = agent.choose(obs)          # 智能体决策
    next_obs, reward, done, info = env.step(action)  # 环境反馈
    agent.store(obs, action, reward, done, next_obs) # 存经验
    obs = next_obs
    if 数据够了: agent.update()          # 用经验更新网络
```

`train_HOPE_ppo.py` 的主循环就是这个结构，只是多了场景选择、RS 规划器接管和日志。

---

## 3. 整体架构

### 3.1 目录与模块关系

```
src/
├── configs.py                  全局配置（几乎所有模块都 from configs import *）
├── env/                        —— 环境 ——
│   ├── car_parking_base.py     CarParking：核心 gym 环境
│   ├── env_wrapper.py          CarParkingWrapper：动作缩放 / 奖励合成 / 观测转置
│   ├── vehicle.py              State、KSModel（运动学自行车模型）、Vehicle、Status
│   ├── lidar_simulator.py      激光雷达仿真
│   ├── observation_processor.py 图像预处理
│   ├── parking_map_normal.py   随机生成 Normal/Complex/Extrem 场景
│   ├── parking_map_dlp.py      读取真实停车场 DLP 数据集场景
│   ├── map_level.py            判定场景难度
│   ├── map_base.py             Area（障碍物容器）
│   └── reeds_shepp.py          Reeds-Shepp 曲线
├── model/                      —— 智能体 ——
│   ├── action_mask.py          动作掩码
│   ├── network.py              MultiObsEmbedding（actor/critic 共用结构）、图像编码器
│   ├── attention.py            Transformer
│   ├── agent_base.py           AgentBase / ConfigBase
│   ├── replay_memory.py        经验缓冲区
│   ├── state_norm.py           观测归一化
│   ├── agent/ppo_agent.py      PPO
│   ├── agent/sac_agent.py      SAC
│   ├── agent/parking_agent.py  混合策略：ParkingAgent + RsPlanner
│   └── ckpt/                   预训练权重（含图像自编码器 autoencoder.pt）
├── train/                      训练脚本（PPO / SAC）
└── evaluation/                 评估脚本与工具
```

依赖关系（箭头表示"使用"）：

```
train_HOPE_ppo.py
   ├─> CarParkingWrapper ─> CarParking ─┬─> ParkingMapNormal / ParkingMapDLP
   │                                    ├─> Vehicle (KSModel)
   │                                    ├─> LidarSimlator
   │                                    ├─> ActionMask.get_steps   (生成观测中的 action_mask)
   │                                    ├─> Obs_Processor          (生成观测中的 img)
   │                                    └─> reeds_shepp            (find_rs_path)
   └─> ParkingAgent ─┬─> PPOAgent ─┬─> MultiObsEmbedding (actor / critic) ─> AttentionNetwork
                     │             ├─> ActionMask.choose_action (按掩码采样动作)
                     │             ├─> StateNorm
                     │             └─> ReplayMemory
                     └─> RsPlanner
```

### 3.2 一个训练 step 的完整数据流

```
             ┌──────────────────────── ParkingAgent.choose_action(obs) ───────────────────────┐
             │ RsPlanner 在执行?                                                               │
             │   是 → action = 下一个 RS 动作；log_prob = PPO 对该动作的对数概率                 │
             │   否 → PPO：obs 归一化 → actor 网络 → 高斯分布(mean, std)                        │
             │          → ActionMask.choose_action：42 个离散动作按 高斯概率×掩码 采样           │
             └────────────────────────────────────────────────────────────────────────────────┘
                                   │ action ∈ [-1,1]^2
                                   ▼
CarParkingWrapper.step: action_rescale → [转角(rad), 速度(m/s)]
                                   ▼
CarParking.step:
   ① 拆成 10 个物理小步，每小步 Vehicle.step；到达就提前停；碰撞就回退这一小步并停下
   ② t += 1；render() 生成观测 {img, lidar, target, action_mask}
   ③ _check_status() → CONTINUE / ARRIVED / OUTBOUND / OUTTIME
   ④ get_reward() → 5 个奖励分量
   ⑤ 若离终点 < 10 m：find_rs_path() 尝试找无碰撞 RS 曲线 → info['path_to_dest']
                                   ▼
CarParkingWrapper: reward_shaping 合成标量奖励；图像 HWC→CHW；done = (status != CONTINUE)
                                   ▼
训练脚本: push_memory(...)；缓冲区满 8192 条 → PPO update()
          若 info['path_to_dest'] 非空 → set_planner_path()，下一步起由 RS 规划器接管
```

---

## 4. 环境设计（`src/env/`）

### 4.1 车辆模型（`vehicle.py`）

- **参考点**：后轴中心。`State` 存 `(x, y, heading, speed, steering)`。
- **车身**：`configs.VehicleBox`，轴距 2.8 m，前悬 0.96 m，后悬 0.93 m，宽 1.94 m，总长约 4.69 m。
- **运动学模型 `KSModel`**（自行车模型）：

  ```
  x'       = v·cos(heading)
  y'       = v·sin(heading)
  heading' = v·tan(steer) / wheel_base
  ```

  动作直接设定**速度**和**转角**（不是加速度），因此速度可以瞬变——这是规划层面的简化。
- **时间尺度**：一个物理小步 0.05 s，一个环境 step = 10 个小步 = 0.5 s。最大速度 2.5 m/s ⇒ 一个 step 最多走 **1.25 m**（这就是 `RsPlanner` 的 `step_ratio`）。
- `Vehicle.step` 返回执行前的状态，`Vehicle.retreat` 可撤销一步——环境用它实现"撞到之前停下"。

### 4.2 场景（地图）

`CarParking.set_level(level)` 决定用哪种地图：

| level | 类 | 说明 |
|---|---|---|
| `Normal` | `ParkingMapNormal` | 随机生成，车位宽敞，障碍少（3 个） |
| `Complex` | `ParkingMapNormal` | 车位更窄，通道更窄，障碍更多（5 个） |
| `Extrem` | `ParkingMapNormal` | **只有侧方停车**，车位长度仅比车长多 0.6~0.9 m，障碍 8 个 |
| `dlp` | `ParkingMapDLP` | 真实停车场数据（Dragon Lake Parking），248 个案例，从 `../data/dlp.data` 读取 |

**随机场景生成**（`generate_bay_parking_case` / `generate_parallel_parking_case`）步骤：

1. 在原点放置目标车位，车位后方是一堵墙；
2. 车位两侧（垂直车位）或前后（侧方车位）随机放"一大块障碍物"或"一排停好的车"；
3. 校验车位宽/长是否落在该难度的 `[MIN, MAX]` 范围内，不满足就递归重来；
4. 通道对面放一堵墙或若干随机多边形障碍；
5. 在通道里随机采一个不碰撞的起点。

地图边界 = 起终点包围盒外扩 10 m（DLP 为 20 m），驶出即 `OUTBOUND`。

DLP 场景还做了数据增强：随机选候选起点并加噪声、各 50% 概率把起点/终点"掉头"。

### 4.3 观测空间

| 键 | 形状 | 含义 | 来源 |
|---|---|---|---|
| `lidar` | (120,) | 120 条射线（每 3°一条，第 0 条朝车头），**车身外缘**到障碍物的距离，最大 10 m | `LidarSimlator.get_observation` |
| `target` | (5,) | `[到终点距离, cos(终点方位角), sin(终点方位角), cos(相对朝向), cos(相对朝向)]`，角度都相对于自车朝向 | `CarParking._get_targt_repr` |
| `img` | (3, 64, 64) | 以自车为中心、车头朝上的鸟瞰图，背景黑、障碍灰、车位绿；由 256×256 截图降采样 4 倍、归一化到 [0,1] | `_get_img_observation` + `Obs_Processor` |
| `action_mask` | (42,) | 42 个离散动作各自可安全执行的比例 [0,1] | `ActionMask.get_steps` |

> 注意：图像观测来自 pygame 渲染，所以即使不需要显示画面，训练时每步也都要渲染（`render_mode='rgb_array'` 会隐藏窗口，但仍渲染）。

观测在送进网络前，`lidar` 和 `target` 还会经过 `StateNorm`（滑动均值/标准差归一化），`img` 和 `action_mask` 不做归一化。

### 4.4 动作空间

- **连续动作**：网络输出 `[steer, speed] ∈ [-1, 1]²`，`action_rescale` 线性映射到 `steer ∈ [-0.75, 0.75] rad`、`speed ∈ [-2.5, 2.5] m/s`（负速度＝倒车）。
- **离散动作集**（用于动作掩码）：`configs.discrete_actions`，转角从 0.75 到 -0.75 每 0.075 取一个（21 个）× 速度 {+1, -1}（前进 / 倒车）= **42 个**。前 21 个是前进，后 21 个是倒车。

### 4.5 step 流程与"碰撞"处理

见 §3.2 的 ①～⑤。需要特别理解的一点：

> **`ENV_COLLIDE = False`，环境实际上不会因碰撞而结束 episode。**
> 如果某个物理小步会撞上障碍物，车辆会回退到这一小步之前并停下，这一 step 剩下的小步不再执行，状态仍是 `CONTINUE`。
> 所以 `Status.COLLIDED` 和对应的 -50 惩罚在默认配置下几乎不会出现；"撞"的代价体现在：这一步没走多远 → 距离奖励少、时间惩罚照扣。真正防止"往障碍物上开"的主要机制是动作掩码。

### 4.6 终止条件

| 状态 | 条件 | 终止奖励（×0.1 前） |
|---|---|---|
| `ARRIVED` | 车身与目标车位重叠面积 > 车位面积的 95% | +50 |
| `OUTBOUND` | 后轴中心驶出地图边界 | -50 |
| `OUTTIME` | 步数 > 200（`TOLERANT_TIME`） | -1 |
| `COLLIDED` | 见 §4.5，默认几乎不触发 | -50 |

### 4.7 奖励设计

每个 `CONTINUE` 步，环境返回 5 个分量（`CarParking._get_reward`），`env_wrapper.reward_shaping` 用 `configs.REWARD_WEIGHT` 加权求和，最后整体 × `REWARD_RATIO = 0.1`：

| 分量 | 权重 | 计算方式 | 作用 |
|---|---|---|---|
| `time_cost` | 1 | `-tanh(t / 2000)` | 每步一点点惩罚，越拖越大，鼓励尽快完成 |
| `rs_dist_reward` | 0（关闭） | RS 路径长度缩短量 | 计算昂贵，默认不用 |
| `dist_reward` | 5 | （上一步到终点距离 − 这一步距离）/ max(起终点距离, 10) | 向终点靠近给正奖励，远离给负奖励 |
| `angle_reward` | 0（关闭） | 朝向偏差缩小量 | 默认不用 |
| `box_union_reward` | 10 | 车身与车位 IoU **刷新历史最高值时**的增量 | 鼓励车身真正进入车位；只奖励"新纪录"，避免来回晃动刷分 |

`dist_reward` 这种"差分形式"的奖励（这一步比上一步好多少）属于**基于势函数的奖励塑形（potential-based reward shaping）**：整个 episode 累加后只取决于起点和终点状态，不会诱导智能体绕圈刷分。`box_union_reward` 则只在 IoU 刷新纪录时给增量，累计值等于整个 episode 达到过的最大 IoU，同样无法靠来回晃动刷分。

---

## 5. 动作掩码（`model/action_mask.py`）

### 5.1 想解决什么

连续动作空间里，大部分随机动作在狭窄车位附近都会"撞"。如果让 RL 自己慢慢学，效率很低。动作掩码在每一步直接告诉智能体："这 42 个动作里，哪些能安全走完、能走多少"。

### 5.2 计算方法

**初始化（只做一次，与环境无关）**：

1. `init_vehicle_box`：假设车在原点，对每个离散动作沿圆弧走 `n_iter = 10` 个小步，记录每一小步的车身四角 → 形状 `(42, 10, 4, 2)`。
2. `precompute`：对每条雷达射线 i、每个动作 j、每个小步 k，计算车身沿射线 i 方向伸出的最远距离 → `dist_star[i, j, k]`。射线方向用线性插值上采样 10 倍（120→1200 条），减少漏检。

**每一步运行时**（`get_steps(lidar)`）：

1. 把雷达读数恢复为"原点到障碍物距离"，同样上采样到 1200 条；
2. 若 `dist_star[i,j,k] <= 雷达距离[i]`，说明动作 j 第 k 小步在射线 i 方向不会撞；
3. 对每个动作找"第一个会撞的小步"，再在所有射线上取最小 → 每个动作最多能安全走几小步；
4. `post_process`：最极端转角各减 1 步，再在相邻转角之间做宽度 5 的最小值滤波（邻居不安全时自己也保守些），除以 10 → `[0, 1]`。

本质上是把"动作是否碰撞"这个昂贵的几何问题，变成了"查表 + 比较"。

### 5.3 如何使用掩码

1. **作为观测输入**：`action_mask` 作为第 4 个 token 输入网络，让网络"看见"安全情况。
2. **约束采样**（PPO 的 `choose_action`）：`ActionMask.choose_action(mean, std, mask)`：
   - 对 42 个离散动作计算在策略高斯分布下的概率密度；
   - 乘以掩码值（不安全的动作概率变 0，只能部分走的动作被降权）；
   - 归一化后随机采样一个，输出归一化动作（转角 /0.75，速度 ±1）。

> 这里的"离散化 + 加权采样"只发生在选动作时；网络本身仍然输出连续高斯分布，PPO 训练时也按连续高斯分布计算概率（见 §13 的说明）。

---

## 6. 网络结构（`model/network.py`、`model/attention.py`）

actor 和 critic 都是 `MultiObsEmbedding`，结构相同，只是输出不同。

```
lidar (120)        ─ Linear→Tanh→Linear ─┐
target (5)         ─ Linear→Tanh→Linear ─┤
action_mask (42)   ─ Linear→Tanh→Linear ─┼─ 堆叠成 4 个 token (B, 4, 128)
img (3×64×64)      ─ 预训练 CNN 编码器(冻结) ─ Tanh→Linear ─┘
                                           │
                     Transformer (1 层, 8 头, 每头 32 维, PreNorm + 残差)
                                           │
                               展平 (B, 512) → Linear(128) → Tanh → Linear(out)
                                           │
          actor：out=2，再经 Tanh → [-1,1]，作为高斯均值       critic：out=1，状态价值 V(s)
```

- **图像编码器**：`ImgEncoder`（2 个卷积块 [4, 8 通道] + 全连接 256 → 128 维）。权重来自预训练自编码器 `model/ckpt/autoencoder.pt`（`AE_Conv`），训练 RL 时**冻结**（`UPDATE_IMG_ENCODE = False`）。这是常用做法：先无监督学好图像特征，再让 RL 只学决策，训练更稳定。
- **为什么用 Transformer**：不同模态信息的重要性随场景变化（例如靠近车位时雷达和掩码更重要）。自注意力让各 token 互相"查询"，自适应地融合。
- **激活函数**：默认全部 Tanh（`use_tanh_activate=True`），PPO 实践中常用。
- **策略的标准差**：不由网络输出，而是一个独立的可学习参数 `log_std`（形状 (1, 2)），与状态无关。训练日志里的 `action_std0/1` 就是它，下降说明策略越来越确定。

---

## 7. PPO 算法实现（`model/agent/ppo_agent.py`）

### 7.1 PPO 的核心思想

策略梯度方法每次更新都在"让好动作概率变大、坏动作概率变小"，但步子太大容易把策略改坏。PPO 用一个**截断的概率比**限制每次更新幅度：

```
ratio   = π_new(a|s) / π_old(a|s)
L_actor = -E[ min( ratio·A,  clip(ratio, 1-ε, 1+ε)·A ) ]      ε = 0.2
L_critic = E[ (V(s) - V_target)^2 ]
```

### 7.2 一次 `update()` 的流程

1. **取数据**：缓冲区里按时间顺序的 8192 条经验 `(s, a, r, done, log_prob_old, s')`（GAE 需要时间顺序，所以不能打乱）。
2. **算优势（GAE，Generalized Advantage Estimation）**：

   ```
   δ_t = r_t + γ·(1-done_t)·V(s_{t+1}) - V(s_t)
   A_t = δ_t + γ·λ·(1-done_t)·A_{t+1}          (从后往前累加，λ = 0.95)
   V_target_t = A_t + V_target_net(s_t)
   A 再做标准化（均值 0、方差 1）
   ```

   λ 在"偏差"和"方差"之间折中：λ=0 只看一步 TD 误差，λ=1 等价于蒙特卡洛回报。
3. **多轮小批量训练**：重复 `mini_epoch = 10` 轮；每轮打乱后按 `mini_batch = 32` 切分（每轮 256 个小批量），对每个小批量计算上面的 actor/critic 损失并更新。每轮结束后用 `TAU = 0.1` 软更新目标 critic。
4. **清空缓冲区**（on-policy：旧数据不再使用）。

### 7.3 其他技巧

| 技巧 | 开关 | 说明 |
|---|---|---|
| 优势标准化 | `adv_norm=True` | 让不同批次的优势尺度一致 |
| 观测归一化 | `state_norm=True` | `StateNorm`，统计量随模型保存 |
| 正交初始化 | `orthogonal_init=True` | PPO 常用初始化 |
| 目标 critic | 软更新 | 让价值目标更稳定 |
| 学习率 | actor 5e-6，critic 2.5e-5 | 很小，配合大 batch |
| 熵正则 / 梯度裁剪 / 奖励归一化 | 默认关闭 | 代码保留了开关 |

### 7.4 `choose_action` vs `get_action`

| 方法 | 采样方式 | 谁在用 |
|---|---|---|
| `choose_action(obs)` | 用动作掩码在 42 个离散动作中采样 | PPO 训练、PPO 评估 |
| `get_action(obs)` | 直接从高斯分布采样（不用掩码采样） | SAC 训练与评估 |
| `get_log_prob(obs, a)` | 只算给定动作的对数概率 | RS 规划器接管时 |

---

## 8. 混合策略：RL + Reeds-Shepp（`parking_agent.py`、`reeds_shepp.py`）

### 8.1 Reeds-Shepp 曲线是什么

对于"能前进也能倒车、有最小转弯半径"的车辆，在**没有障碍物**时，任意两个位姿之间的最短路径一定由不超过 5 段的 **L（左转圆弧）/ R（右转圆弧）/ S（直线）** 组成，每段可以前进或倒车。共有 48 种组合，`reeds_shepp.py` 枚举所有可行组合并给出路径。可以把它当黑盒：

```python
paths = calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size)  # 所有候选
path  = calc_optimal_path(...)                                       # 最短的一条
# path.ctypes = ['L','S','R'], path.lengths = [1.2, -3.0, 0.8]（米，负数是倒车）
```

### 8.2 环境端：找一条无碰撞曲线（`CarParking.find_rs_path`）

每一步若离终点 < `RS_MAX_DIST = 10 m`：

1. 计算当前位姿到车位的全部 RS 候选路径；
2. 按长度从短到长依次用 `is_traj_valid` 检查（车身在每个轨迹点上是否与障碍物边相交、是否出界）；
3. 返回第一条无碰撞的路径；路径超过最短长度 1.6 倍时放弃。

找到就放进 `info['path_to_dest']`。

### 8.3 智能体端：接管执行（`ParkingAgent` + `RsPlanner`）

- 训练/评估脚本看到 `info['path_to_dest']` 后调用 `parking_agent.set_planner_path(path)`。只有规划器空闲时才接受，所以一旦开始执行就会执行完。
- `RsPlanner.set_rs_path` 把每一段换成动作：L/R/S → 转角 1/-1/0（满舵左/满舵右/直行），长度除以 1.25 m 得到步数，每步速度 ±1，余量用小于 1 的速度。
- 执行期间，`ParkingAgent.choose_action` 返回规划器的动作，同时调用 PPO 的 `get_log_prob` 为该动作算对数概率，这些经验**也会存进 PPO 的缓冲区参与训练**——相当于让 RL 也"看到"几何规划器如何完成最后一段。
- 动作队列用完自动 reset，控制权交还 RL。

为什么有效：RL 擅长在复杂环境中"挪到一个好位置"，但要精确停进车位很难（奖励稀疏、精度要求高）；RS 曲线在近距离、无碰撞时是精确最优解。两者互补。

---

## 9. 训练流程（`train/train_HOPE_ppo.py`）

### 9.1 运行方式

```bash
cd src                                         # 必须在 src 下运行（代码用了相对路径）
python ./train/train_HOPE_ppo.py --visualize "" # 见 §13：传空字符串才能关掉可视化
tensorboard --logdir ./log/exp/                # 查看训练曲线
```

参数：`--agent_ckpt` 继续训练的模型、`--img_ckpt` 图像编码器（默认 `./model/ckpt/autoencoder.pt`）、`--train_episode` 训练 episode 数（默认 10 万）、`--eval_episode` 训练结束后的评估数量。

### 9.2 初始化

1. 创建 `CarParking` → `CarParkingWrapper`；
2. 创建 `SceneChoose`（选场景类型）和 `DlpCaseChoose`（选 DLP 案例）；
3. 日志目录 `./log/exp/ppo_<时间戳>/`，并把 `configs.py` 拷贝进去备份；
4. 固定随机种子 `SEED = 42`；
5. 创建 `PPOAgent`，可选加载已有模型，加载预训练图像编码器；
6. 组合成 `ParkingAgent(rl_agent, RsPlanner(1.25))`。

### 9.3 主循环（每个 episode）

1. **选场景（课程式采样）**：
   - `SceneChoose`：前 200 个 episode 在 4 类场景间均衡轮换；之后 50% 概率按"目标成功率（0.95/0.95/0.9/0.99）− 近期成功率"加权，挑表现最差的场景练，50% 继续均衡；
   - 若选中 `dlp`，再由 `DlpCaseChoose` 按各案例的失败率加权选具体案例。
2. **交互**：`choose_action → env.step → push_memory`；缓冲区达到 8192 条就 `update()`。
3. **规划器接管**：`info['path_to_dest']` 非空 → `set_planner_path`。
4. **记录结果**：成功/失败回报给两个采样器。
5. **日志**：TensorBoard 记录 `total_reward`、`avg_reward`、`action_std0/1`（实际是 log_std）、各场景成功率、`step_num`、actor/critic loss。
6. **保存模型**：四类场景近 100 次成功率**同时**不低于历史最佳（最佳值以目标成功率为上限）时保存 `PPO_best.pt`；每 2000 个 episode 保存一次检查点。

### 9.4 训练结束

依次在 dlp / Extrem / Complex / Normal 上调用 `eval()`，结果写到日志目录的对应子目录。

---

## 10. SAC 版本的差异（`sac_agent.py`、`train_HOPE_sac.py`）

| 方面 | PPO | SAC |
|---|---|---|
| 类型 | on-policy | off-policy |
| 数据 | 攒 8192 条，更新完即清空 | 回放池 10240 条，随机采样 32 条，可反复使用 |
| 更新频率 | 缓冲区满一次 | 前 10240 步纯随机探索，之后每 10 步更新一次 |
| critic | V(s)，1 个 + 目标网络 | Q(s,a)，2 个取最小 + 各自目标网络（动作作为第 5 个 token 输入） |
| 探索 | 高斯噪声 + 掩码采样 | 最大熵目标，熵权重 alpha 自动调节 |
| 选动作 | `choose_action`（掩码采样） | `get_action`（直接高斯采样，掩码只作为输入） |

SAC 更新公式（`SACAgent.update`）：

```
y        = r + γ(1-done)·[ min(Q1', Q2')(s', a') - α·log π(a'|s') ],  a' ~ π(·|s')
L_Q      = (Q_i(s,a) - y)^2
L_actor  = α·log π(a|s) - min(Q1, Q2)(s, a),  a 用重参数化采样
L_alpha  = α·(-log π(a|s) - target_entropy)
```

---

## 11. 评估（`evaluation/`）

```bash
cd src
python ./evaluation/eval_mix_scene.py ./model/ckpt/HOPE_SAC0.pt --eval_episode 10 --visualize True
```

- 文件名含 `ppo` 的按 PPO 加载并使用掩码采样，否则按 SAC 加载。
- `eval_utils.eval` 第 i 个 episode 使用 `case_id = i+1`（DLP 会按编号遍历全部案例）。
- **脱困技巧**：若连续两步 `target` 观测完全一样（车被卡住），改为随机动作。
- 输出：总体成功率、平均奖励、（DLP 时）按难度的成功率；保存 `success_rate.png`（步数→累计成功率）、`record.data`（每个 episode 的详细记录）、`result.txt`。

---

## 12. 关键超参数一览（`configs.py`）

| 类别 | 参数 | 值 | 含义 |
|---|---|---|---|
| 车辆 | `WHEEL_BASE / WIDTH / LENGTH` | 2.8 / 1.94 / ≈4.69 m | 车辆尺寸 |
| | `VALID_STEER` | ±0.75 rad | 最大前轮转角 |
| | `VALID_SPEED` | ±2.5 m/s | 最大速度 |
| | `NUM_STEP × STEP_LENGTH` | 10 × 0.05 s | 一个环境 step 的时长 |
| 环境 | `TOLERANT_TIME` | 200 | 最大步数 |
| | `LIDAR_NUM / LIDAR_RANGE` | 120 / 10 m | 激光雷达 |
| | `OBS_W×OBS_H` | 256×256 → 64×64 | 图像观测 |
| | `RS_MAX_DIST` | 10 m | 离终点多近开始尝试 RS 曲线 |
| | `ENV_COLLIDE` | False | 碰撞不终止 episode |
| 奖励 | `REWARD_RATIO` | 0.1 | 总奖励缩放 |
| | `REWARD_WEIGHT` | time 1, dist 5, box_union 10 | 各分量权重 |
| 模型 | `GAMMA` | 0.98 | 折扣因子 |
| | `BATCH_SIZE` | 8192 | PPO 每次更新的数据量 |
| | `LR` | 5e-6 | 基础学习率 |
| | `TAU` | 0.1 | 目标网络软更新系数（PPO） |
| | `ATTENTION_CONFIG` | 1 层，8 头，每头 32 维 | Transformer |
| | `USE_IMG / USE_ACTION_MASK / USE_ATTENTION` | True | 模态与结构开关 |
| PPO | `mini_epoch / mini_batch` | 10 / 32 | 每批数据训练轮数 / 小批量大小 |
| | `clip_epsilon / lambda_` | 0.2 / 0.95 | PPO 截断范围 / GAE λ |

`MAP_LEVEL` 只决定环境创建时的默认场景，训练时每个 episode 都会通过 `env.reset(..., level)` 重新指定。

---

## 13. 阅读时需要注意的地方

下面是读代码时发现的、容易让初学者困惑或疑似有问题的地方。**我没有修改任何代码逻辑**——尤其是会影响观测/网络的地方，改动会导致现有预训练权重失效。

### 13.1 疑似 bug

1. **`target` 观测的第 5 维**（`car_parking_base.py` `_get_targt_repr`）：注释说是 `sin(相对朝向)`，代码写的是两次 `cos(rel_dest_heading)`。后果：网络无法区分目标朝向偏左还是偏右（只能从图像和其他信息推断）。预训练模型是在这个版本下训练的，若要修正，需要重新训练。
2. **`--visualize` / `--verbose` 参数**：`argparse` 的 `type=bool` 会把任何非空字符串（包括 `"False"`）转成 `True`。所以 `--visualize False` **仍然会弹窗**；要关闭需传空字符串 `--visualize ""`，或者直接修改默认值。
3. **正交初始化的输出层 gain**（`network.py`）：本意是最后一层用 gain=0.01，但循环里的计数变量 `i` 从未递增，所以所有层都用 gain=1。
4. **未启用的代码路径有错误**：`PPOAgent.update` 里的学习率衰减（`LR_DECAY=False` 时不执行）、`AgentBase.load` / `PPOAgent.load` 的非 `params_only` 分支。默认流程不会走到。

### 13.2 设计上的近似（不是 bug，但要知道）

1. **碰撞不终止 episode**（§4.5）。卡住的车只能等超时；评估时用随机动作帮助脱困，训练时没有。
2. **PPO 的 `log_prob` 是近似的**：用动作掩码采样时，实际选动作的概率是"高斯密度×掩码再归一化"，但存下的 `log_prob` 是该动作在**原始高斯分布**下的对数概率；PPO 更新时也用高斯分布计算新概率。因此概率比 `ratio` 是一个近似。
3. **RS 规划器产生的数据进入了 PPO 缓冲区**：这些动作不是当前策略采样出来的，严格说不符合 on-policy 假设，但在实践中起到了类似"示范数据"的作用。
4. **`StateNorm` 的统计量在训练过程中持续变化**：同一个原始观测在不同时间被归一化成不同的值。保存/加载模型时务必连同 `state_norm` 一起（代码已经这么做了）。

### 13.3 运行环境相关

- 所有脚本都需要在 `src/` 目录下运行（`../data/dlp.data`、`./configs.py`、`./model/ckpt/...` 都是相对路径）。
- `pygame` 是必需依赖：观测图像来自渲染。无显示器的服务器上可以在 `configs.py` 里取消 `os.environ["SDL_VIDEODRIVER"]="dummy"` 的注释。

---

## 14. 建议的上手实验

1. **先跑评估看效果**：用 `HOPE_SAC0.pt` 跑 10 个 episode，带可视化，观察 RL 段和 RS 段的切换（离车位约 10 m 内路径会突然变得很"规整"）。
2. **单步调试环境**：在 Python 里创建 `CarParkingWrapper(CarParking(render_mode='rgb_array'))`，`reset()` 后打印各观测的 shape 和数值，手动 `step([0, 1])` 看奖励分量。
3. **可视化动作掩码**：在某个观测上画出 `obs['action_mask']` 的 42 个值（前 21 个前进、后 21 个倒车），对照画面理解哪些方向被禁止。
4. **消融实验**：在 `configs.py` 中分别关闭 `USE_ACTION_MASK`、`USE_IMG`、`USE_ATTENTION`，或把 `RS_MAX_DIST` 设为 0（禁用混合策略），比较 TensorBoard 中的成功率曲线。注意：改动观测/网络结构后不能再加载原有预训练权重。
