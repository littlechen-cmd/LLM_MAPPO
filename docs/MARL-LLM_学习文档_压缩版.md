# MARL-LLM (LAMARL) 项目知识压缩文档

> **论文**: LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation
> **发表**: IEEE RA-L 2025, Vol.10, No.7, pp.7476-7483
> **任务**: 多智能体协同组装——智能体集群通过协作运动填充目标形状区域（覆盖率+均匀度最大化）
> **核心创新**: LLM生成奖励函数&先验策略 → 引导MADDPG训练 → 知识从LLM迁移到RL

---

## 1. 总体架构

```
                    目标形状图像(.png)
                          │
                    ┌─────▼─────┐
                    │ 图像预处理  │ cfg/assembly_cfg.py
                    │ PNG→网格坐标│ → results.pkl
                    └─────┬─────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
┌─────────┐    ┌──────────────────┐    ┌──────────────┐
│ LLM模块  │    │   MARL训练引擎    │    │  C++物理引擎   │
│          │    │                  │    │              │
│ GPT API  │───▶│ 奖励函数+先验策略  │    │ 观测构建       │
│ 代码生成  │    │  ↓               │◄───│ 碰撞检测       │
│ 代码审查  │    │ MADDPG/AIRL训练  │    │ 奖励计算       │
└─────────┘    └────────┬─────────┘    └──────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │   评估 & 可视化    │
              │ 覆盖率/均匀度/     │
              │ Voronoi均匀度     │
              └──────────────────┘
```

**四种训练方法(training_method)**:
- `manual_rl`: 纯MADDPG，环境稀疏奖励
- `llm_rl`: LLM生成先验策略 → MADDPG训练时MSE正则化引导
- `irl`: AIRL逆强化学习（从专家演示学奖励函数）
- `pid`: PID控制器（规则baseline）

---

## 2. 全部配置参数

```python
# === 文件 cfg/assembly_cfg.py ===
# 智能体系统
--n_a=30                  # 智能体数量
--is_boundary=True        # 墙壁边界(True)/周期边界(False)
--dynamics_mode='Cartesian' # 动力学模型
--is_feature_norm=False   # 特征归一化

# 可视化
--render_traj=True        # 渲染轨迹
--traj_len=15             # 轨迹历史长度

# 智能体行为
--agent_strategy='input'  # input/random/llm/rule
--training_method='llm_rl' # llm_rl/pid/manual_rl/irl
--is_collected=False      # 是否采集专家数据

# === 训练超参数 ===
--env_name='assembly'
--seed=226
--n_rollout_threads=1     # 并行环境线程
--n_training_threads=5    # CPU训练线程
--buffer_length=20000     # 回放缓冲区容量
--n_episodes=3000         # 总训练回合数
--episode_length=200      # 每回合最大步数
--batch_size=512          # 网络更新批大小
--hidden_dim=180          # 隐藏层维度
--lr_actor=1e-4           # Actor学习率
--lr_critic=1e-3          # Critic学习率
--epsilon=0.1             # epsilon-greedy探索
--noise_scale=0.9         # 动作噪声缩放
--tau=0.01                # 目标网络软更新率
--agent_alg='MADDPG'      # MADDPG/DDPG
--device='cpu'            # cpu/gpu
--save_interval=10        # 检查点保存间隔
--gamma=0.95              # 折扣因子 (hardcoded)

# IRL专用
--lr_discriminator=1e-3
--disc_use_linear_lr_decay=False
--hidden_num=4            # 判别器隐藏层数

# === 图像预处理 ===
# process_image(): PNG→二值化→36px网格化→中心化→缩放到target_height=2.2
# 输出: results.pkl = {l_cell, grid_coords, binary_image, shape_bound_points}
```

---

## 3. 神经网络架构

### 3.1 MLPNetwork (Actor/Critic共用)
```python
class MLPNetwork(nn.Module):
    # input_dim → fc(→hidden_dim) → LeakyReLU → fc(→hidden_dim) → LeakyReLU
    #          → fc(→hidden_dim) → LeakyReLU → fc(→out_dim)
    # Actor: output→tanh (constrain_out=True), 输出范围[-1,1]
    # Critic: output→linear (constrain_out=False), 输出单个Q值
```

### 3.2 AIRL判别器网络
```python
class Discriminator(nn.Module):
    # 包含两个MLPUnit子网络:
    #   g(s,a):  state+action → 1  (奖励近似)
    #   h(s):    state → 1         (势函数)
    # 前向: f(s,a,s') = g(s,a) + γ·h(s') - h(s)
    # 最终输出: f(s,a,s') - log_π(a|s)
    # 奖励计算: reward = g(s,a) + γ·h(s') - h(s)

class MLPUnit(nn.Module):
    # input → [Linear→LeakyReLU] × hidden_num → Linear(→1)
```

### 3.3 残差块（MLPNetworkRew用）
```python
class ResidualBlock(nn.Module):
    # x → fc→LeakyReLU→fc → +x(residual) → LeakyReLU
```

---

## 4. 核心算法

### 4.1 DDPGAgent
```python
class DDPGAgent:
    def __init__(self, dim_input_policy, dim_output_policy, dim_input_critic, ...):
        # 四个网络: policy, target_policy, critic, target_critic
        # Actor: MLPNetwork(dim_input_policy→dim_output_policy, tanh输出)
        # Critic: MLPNetwork(dim_input_critic→1)
        # 硬拷贝初始化: target_policy←policy, target_critic←critic
        # 优化器: Adam(policy, lr_actor), Adam(critic, lr_critic)
        # 探索: GaussianNoise(连续) / epsilon-greedy(离散)

    def step(self, obs, explore=False):
        action = self.policy(obs)  # tanh输出[-1,1]
        if explore:
            if rand() < epsilon:
                action = uniform(-1, 1)           # 随机探索
            else:
                action += GaussianNoise(scale)    # 加噪声探索
            action = clamp(action, -1, 1)
        return action, log_pi
```

### 4.2 MADDPG核心更新 (双网络架构)
```python
class MADDPG:
    def update(self, obs, acs, rews, next_obs, dones, agent_i, acs_prior=None, alpha=0.5):
        agent = self.agents[agent_i]

        # === Critic更新 (TD误差最小化) ===
        target_acs = all agents' target_policy(next_obs)  # 集中式Critic
        target_Q = rews + γ * agent.target_critic([next_obs, target_acs]) * (1-dones)
        current_Q = agent.critic([obs, acs])
        critic_loss = MSE(current_Q, target_Q.detach())

        # === Actor更新 (最大化Q值+先验正则化) ===
        curr_actions = agent.policy(obs)
        Q_val = agent.critic([obs, curr_actions])
        actor_loss = -Q_val.mean()  # 梯度上升

        # LLM先验正则化 (核心创新点)
        if acs_prior is not None:
            mask = (acs_prior.abs() >= 1e-2).any(dim=1)  # 过滤无效先验
            reg_loss = MSE(curr_actions[mask], acs_prior[mask])
            actor_loss += 0.3 * alpha * reg_loss

        # 目标网络软更新(每次update_all_targets调用):
        # θ_target = τ·θ + (1-τ)·θ_target  其中τ=0.01

# 初始化和保存/加载
MADDPG.init_from_env(env, ...)   # 从环境维度自动构建
MADDPG.init_from_save(filename)  # 从检查点加载
MADDPG.save(filename)            # 保存所有agent参数
```

### 4.3 AIRL逆强化学习
```python
class AIRL:
    def update(self, states, actions, log_pis, next_states, dones):
        # 采样专家数据 (batch_size×6)
        states_exp, actions_exp, ... = expert_buffer.sample(6*batch_size)

        # 判别器损失 (最小化负对数sigmoid)
        logits_pi = discriminator(states, actions, log_pis, next_states, dones)
        logits_exp = discriminator(states_exp, actions_exp, log_pis_exp, ...)
        loss = -logsigmoid(-logits_pi).mean() - logsigmoid(logits_exp).mean()

    def calculate_reward(self, states, actions, log_pis, next_states, dones):
        # 奖励 = g(s,a) + γ·h(s') - h(s)  (AIRL的恢复奖励)
        return self.discriminator.f(states, actions, next_states, dones)
```

### 4.4 探索噪声
```python
class GaussianNoise:
    def noise(self, N):      return randn(N, dim) * scale
    def log_prob(self, noises): return -0.5 * Σ(noise/scale)² - dim*log(scale*√(2π))

# 训练时噪声衰减: noise = max(0.5, noise - noise_scale/n_episodes)
# llm_rl模式: noise从0.9→0.5; irl模式: noise从0.9→0.4
```

---

## 5. 训练流程

### 5.1 MADDPG (manual_rl / llm_rl)
```
每个episode:
  1. env.reset() → obs
  2. maddpg.prep_rollouts(device='cpu')
  3. 设置探索噪声
  4. for t in range(episode_length=200):
       actions = maddpg.step(obs, explore=True)    # 含噪声探索
       next_obs, rewards, dones, _, ac_prior = env.step(actions)
       buffer.push(obs, actions, rewards, next_obs, dones, ac_prior)
       obs = next_obs
  5. maddpg.prep_training(device)
  6. for _ in range(20):                           # 每episode更新20轮
       for agent_i in range(n_agents):
         sample = buffer.sample(batch_size=512)
         maddpg.update(sample, agent_i, acs_prior, alpha)
         maddpg.update_all_targets()               # 软更新目标网络
  7. maddpg.prep_rollouts(device='cpu')
  8. noise衰减, alpha=0.1
  9. 每10episode: 记录TensorBoard日志
  10. 每40episode: 保存增量检查点
```

### 5.2 AIRL (irl)
```
训练交替:
  MADDPG rollout (同5.1) + 记录log_pi到buffer
  ↓
  每3个episode:
    for _ in range(20):
      采样policy数据 + 采样expert数据
      airl.update()  # 训练判别器
  ↓
  for _ in range(20):
    采样policy数据
    rewards = airl.discriminator.calculate_reward()  # 判别器奖励替代环境奖励
    maddpg.update(rewards=rewards)                   # 用IRL奖励训练策略
```

### 5.3 回放缓冲区
```python
class ReplayBufferAgent:
    # 循环缓冲区: max_steps × num_agents 容量
    # push(): 存储 (obs, action, reward, next_obs, done, [ac_prior], [log_pi])
    # sample(N): 随机采样N条经验 → Tensor(可选GPU)
    # 采样范围: [random(0, 300000), total_length-300000+random]  # 避开边界
```

---

## 6. 环境规范

### 6.1 AssemblySwarmEnv (cus_gym)
```
物理参数:
  智能体半径: 0.035, 感知半径: 3.0, 避碰半径: 0.15
  速度范围: [0, 0.8], 加速度范围: [-1, 1]
  场地: 4.8×4.8 (boundary_half=2.4)
  Δt: 0.1, 帧率: 45fps
  碰撞刚度: k_ball=30(N/m), k_wall=100(N/m), c_wall=5(N/m/s)

观察空间 (2D连续):
  - 自身位置(2) + 速度(2)
  - 最近6个邻居的相对位置(12) + 邻居速度(12)  [topo_nei_max=6]
  - 目标形状信息: 最近目标网格相对位置(2) + 网格是否被占据(1)
  - 总共 ~31维 (含壁障信息)

动作空间 (2D连续):
  - 2维加速度 [ax, ay], 范围 [-1, 1]

奖励函数 (C++实现):
  - 稀疏奖励: 智能体到达目标网格时获得正奖励
  - 惩罚项: 碰撞惩罚、探索惩罚

C++加速层:
  cus_gym/gym/envs/customized_envs/envs_cplus/
  ├── AssemblyEnv.cpp/h    # 观测构建、碰撞检测、奖励计算
  ├── c_lib.py             # ctypes绑定
  └── CMakeLists.txt       # 编译→libAssemblyEnv.so
```

### 6.2 评估指标 (AssemblySwarmWrapper)
```python
# coverage_rate(): 被智能体占据的网格数 / 总目标网格数
#   occupied if ||agent_pos - grid_center|| < r_avoid/2

# distribution_uniformity(): 智能体间最小距离的归一化方差
#   uniform = Var(min_dist_i for i in agents)
#   metric = (uniform - min(min_dist)) / (max(min_dist) - min(min_dist))

# voronoi_based_uniformity(): 每个智能体Voronoi区域内网格数的归一化方差
#   每个网格分配给最近的智能体 → 计算每智能体网格数的方差 → 归一化
```

---

## 7. LLM集成机制

### 7.1 LLM生成流程
```
rl_generate_functions.py (入口)
  │
  ├── RLGeneration (生成阶段)
  │   ├── 构建prompt: 环境描述 + robot_api + 任务要求
  │   ├── 调用GPT API (异步, gpt.py)
  │   │   └── 重试: 5次, 指数退避, 流式输出
  │   └── 解析: 从LLM响应中提取Python代码→compute_reward + robot_policy函数
  │
  └── RLCodeReview (审查阶段)
      ├── 检查语法错误
      ├── 验证函数签名
      ├── 检查API调用合法性
      └── 反馈LLM进行修正
```

### 7.2 LLM生成的产物
```python
# 1. 奖励函数 compute_reward(state_dict) → float
#    比环境稀疏奖励更密集的奖励信号

# 2. 先验策略 robot_policy(obs) → action
#    作为训练中的行为先验, 通过MSE正则化引导RL

# 使用方式 (训练时):
# env.step()返回 ac_prior = robot_policy(obs)
# MADDPG.update()中: actor_loss += 0.3*alpha*MSE(curr_action, ac_prior)
```

### 7.3 LLM配置
```yaml
# llm/config/llm_config.yaml
api_base: {GPT, VLM, QWEN}    # API端点
api_key:  {GPT, VLM, QWEN, CLAUDE}
model: {VLM: gpt-4o, GPT: o1-preview}

# 实验任务: llm/config/experiment_config.yaml → --run_experiment_name: bridging
```

### 7.4 Robot API (LLM可调用的函数)
```
base:        get_neighbor_id_list, get_robot_position_and_velocity
assembly专用: get_unoccupied_cells_position, get_target_cell_position,
             is_within_target_region
```

---

## 8. 评估流程

```python
# eval_assembly.py
1. 加载模型: MADDPG.init_from_save(model.pt)
2. 加载形状: results.pkl → l_cells, grid_center_origins, binary_images, ...
3. 运行episode (length=300):
   - maddpg.prep_rollouts(), noise=0 (确定性)
   - 每300步切换形状 (支持多形状测试)
   - 每步记录: coverage_rate, distribution_uniformity, voronoi_uniformity
4. 保存: metrics.pkl, state_data.npz, 训练曲线图(reward/loss/accuracy)
5. 生成: reward_curve.pdf, loss.pdf, loss_discriminator.pdf, accuracy.pdf
```

---

## 9. 文件组织映射

```
MARL-LLM-master/
├── README.md                         # 安装&使用说明
├── fig/                              # 论文图片(7张PNG) + results.pkl
│
├── marl_llm/                         # ★主代码模块
│   ├── requirements.txt              # Python依赖
│   ├── cfg/
│   │   └── assembly_cfg.py           # ★配置中心: 图像处理+argparse所有参数
│   ├── algorithm/
│   │   ├── algorithms/
│   │   │   ├── maddpg.py             # ★MADDPG完整实现(308行)
│   │   │   └── airl.py               # ★AIRL判别器实现(158行)
│   │   └── utils/
│   │       ├── agents.py             # ★DDPGAgent(119行)
│   │       ├── buffer_agent.py       # ★ReplayBufferAgent(204行)
│   │       ├── buffer_expert.py      # 专家数据缓冲(复用结构)
│   │       ├── buffer_episode.py     # 回合轨迹存储
│   │       ├── networks.py           # ★MLP/残差/判别器网络(174行)
│   │       ├── noise.py              # Gaussian+OU噪声(41行)
│   │       └── misc.py               # soft/hard_update, gumbel_softmax
│   ├── train/
│   │   ├── train_assembly.py         # ★MADDPG训练入口(180行)
│   │   └── train_assembly_airl.py    # ★AIRL训练入口(235行)
│   ├── eval/
│   │   ├── eval_assembly.py          # ★评估入口(300行)
│   │   └── collect_expert_data.py    # 专家数据采集
│   └── llm/                          # LLM交互子系统
│       ├── config/
│       │   ├── llm_config.yaml       # API密钥配置
│       │   └── experiment_config.yaml# 实验任务名
│       └── modules/
│           ├── llm/
│           │   ├── gpt.py            # OpenAI异步客户端(重试+流式)
│           │   └── model_manager.py  # 单例API管理器
│           ├── framework/
│           │   ├── action.py         # ActionNode+ActionLinkedList工作流
│           │   └── actions/
│           │       ├── rl_generate_functions.py  # ★LLM生成主流程
│           │       ├── rl_analyze_generation.py  # 奖励/策略函数生成
│           │       └── rl_code_review.py         # 代码审查验证
│           └── prompt/
│               ├── env_descriptions.py      # 环境文本描述
│               ├── robot_api_prompt.py      # ★Robot API定义(7个函数)
│               ├── rl_generation_prompt.py  # 生成提示模板
│               └── rl_code_review_prompt.py # 审查提示模板
│
└── cus_gym/                          # ★自定义Gym环境
    ├── setup.py                      # pip install -e .
    ├── gym/
    │   ├── envs/customized_envs/
    │   │   ├── assembly.py           # ★AssemblySwarmEnv(~945行)
    │   │   ├── VideoWriter.py        # 视频录制
    │   │   └── envs_cplus/           # C++加速引擎
    │   │       ├── AssemblyEnv.cpp   # 核心物理仿真
    │   │       ├── AssemblyEnv.h
    │   │       ├── c_lib.py          # ctypes Python绑定
    │   │       ├── CMakeLists.txt    # 编译→libAssemblyEnv.so
    │   │       └── build.sh
    │   └── wrappers/customized_envs/
    │       └── assembly_wrapper.py   # ★AssemblySwarmWrapper(评估指标)
    └── spaces/wrappers/utils/        # 标准Gym框架代码(非核心)
```

★ 标记为核心文件, 共约13个关键文件需重点关注

---

## 10. 关键设计决策与代码模式

### 10.1 设备管理策略
```
Rollout阶段: 所有网络必须CPU (C++环境交互+NumPy)
Training阶段: 网络移动到GPU/CPU
prep_rollouts('cpu'): policy.eval(), 仅移动policy到CPU
prep_training('gpu'): 所有4个网络.train(), 全部移到GPU
```

### 10.2 多智能体数据流
```
obs: shape=[obs_dim, n_agents]    (列主序, 每列一个智能体)
acs: shape=[act_dim, n_agents]

buffer存储时转置: observations_orig[:, index].T → shape=[n_agents, obs_dim]
MADDPG.step时: observations[:, start_stop_num[i]].t() → 每个agent独立处理
```

### 10.3 先验正则化细节
```
alpha初始=0.5 (训练循环中动态设为0.1)
reg_weight = 0.3 × alpha = 0.03 (最终)
过滤条件: acs_prior绝对值 < 1e-2 的全零行被忽略(视为LLM未生成该样本的先验)
```

### 10.4 关键依赖
```
torch==2.1.0, gym (with cus_gym), tensorboardX, numpy, opencv-python,
matplotlib, scipy, pyyaml, openai (for GPT API), tqdm, asyncio
```

---

## 11. 常见扩展/修改点

| 修改目标 | 涉及文件 | 关键参数/位置 |
|---------|----------|-------------|
| 调整智能体数量 | assembly_cfg.py | `--n_a` |
| 切换训练方法 | assembly_cfg.py | `--training_method` |
| 修改网络结构 | networks.py | `MLPNetwork`, hidden_dim层数 |
| 调整先验正则化强度 | maddpg.py:174 | `0.3 * alpha * regularization_term` |
| 修改环境奖励 | AssemblyEnv.cpp | C++ compute_reward |
| 替换LLM模型 | llm_config.yaml | model字段 |
| 添加新形状 | fig/目录 | 放PNG → 运行assembly_cfg.py预处理 |
| 修改探索策略 | train_assembly.py:140 | noise衰减公式 |
| AIRL判别器更新频率 | train_assembly_airl.py:157 | `ep_index % 3 == 0` |
