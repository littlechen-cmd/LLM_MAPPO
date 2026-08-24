# 8.4 货物动态入库落地方案(Phase 3)

> 目标:把总体方案 §8.4 的"货物动态入库逻辑"从当前训练特例(一次性入库)切换为正式动态机制,并说明对训练、蒸馏、评估各环节的影响与改动点。
>
> 定位:本方案为 **Phase 3 落地文档**——动态入库在 Phase 3 阶段实施,完成 Phase 3 动态环境下的训练与行为级评估;LLM 接入、特训注入属 Phase 4 后续工作,仅作衔接说明。
>
> 状态:**决策已确认(2026-08-11)**,待实施。涉及核心训练/环境代码的改动需核心工程师授权后实施。

---

## 1. 背景与目标

总体方案 §8.4 定义的最终入库逻辑:

- **入库周期**:每隔固定 `s` steps 触发一次入库事件
- **入库数量**:每次 `n` 个货物,`n ∈ [n_min, n_max]` 均匀采样
- **批次定义**:同一次入库事件到达的货物 = 一个**批次**
- **优先级分配**:
  - 同批内所有货物同优先级
  - 默认优先级按入库时间排序:越早入库的批次优先级越高
  - 字母标签体现排序位置(A 为最早入库,字母表顺序递降)
- **奖励关联**:搬运优先级越高的货物的 Agent 获得越高的完成奖励

当前 Phase 2/3 训练采用的是 **8.4 的退化特例**:整局只入库一次、每批 1 件、批次序号即优先级。它验证了双头架构与蒸馏通道可行,但**未训练"任务中途持续到达"下的重规划与协调能力**。

---

## 2. 现状盘点:能力"已有、被关、缺失"三分类

### 2.1 已实现且可直接复用(环境层)

| 能力 | 位置 | 说明 |
|---|---|---|
| 周期入库调度 | `DynamicWarehouse._spawn_scheduled_batch` | 每 `batch_interval` 步触发,数量按 `batch_size_range` 采样 |
| 批次创建 | `_create_batch` | 支持 `priority_schedule` 轮转 / 动态字母扩展两种模式 |
| 批内同优先级 | `task_queue.create_batch` | 同一批共用同一字母 |
| 动态归一权重 | `TaskQueue.priority_weight` | 按当前活跃字母集合归一化,不依赖固定档位 |
| 优先级奖励关联 | `_complete_delivered_tasks` | `reward = 5.0 * priority_weight` |
| 首批即配置 | Phase 1 曾启用 | `configs/phase1_medium_3ag.yaml`:`batch_interval: 20`、`batch_size_range: [1, 3]`、`labels: dynamic_fifo` |

### 2.2 被 Phase 2/3 人为关闭(需重新打开)

`Phase2Warehouse.__post_init__` 硬编码关闭了动态入库:

```100:108:llm_mappo/phase2.py
        make_options = {
            "disable_env_checker": True,
            "n_agents": self.n_agents,
            "request_queue_size": self.n_agents,
            "max_steps": self.max_steps,
            "batch_interval": self.max_steps + 1,
            "batch_size_range": (1, 1),
            "initial_priority_label": "B",
        }
```

- `batch_interval = max_steps + 1`(401 > 400)→ 整局 `_spawn_scheduled_batch` 永不触发
- `batch_size_range = (1, 1)` → 即使触发也每批 1 件
- reset 时对每个货架单独成批(见 `environment.py` reset 分支)→ 3 件货得到 A1/B1/C1

### 2.3 需要新写或适配的部分

| 缺口 | 说明 |
|---|---|
| **训练侧参数透传** | Phase 2/3 训练从 config 读不到 `batch_interval`/`batch_size_range`,需加配置项并透传 |
| **优先级字母扩展** | `_create_batch` 动态模式从 `initial_priority_label` 起(Phase 2 写死 "B"),与 §8.4"最早批次为 A"不符;且 `priority_schedule=[A,B,C]` 轮转会在第 4 批回到 A,破坏"越早越高"语义 |
| **蒸馏标签映射** | `_engagement_targets` 只认 A/B/else 三档,动态字母表(D、E…)会全部落到 0.3,语义混叠 |
| **特训场景注入** | 三种关键行为(窄通道让行、抢让、电量弃任务)需场景偏置采样,当前缺失;计划 **Phase 4 实施**,Phase 3 不做 |
| **行为级评估** | 整体指标会被平凡场景稀释,需按行为分组验证 |

---

## 3. 关键代码事实(改动依据)

### 3.1 周期入库触发逻辑

```245:267:llm_mappo/environment.py
    def _spawn_scheduled_batch(self):
        if self._cur_steps == 0 or self._cur_steps % self.batch_interval:
            return
        active_shelf_ids = {task.shelf_id for task in self.task_queue.active_tasks}
        carried_shelf_ids = {
            agent.carrying_shelf.id
            for agent in self.agents
            if agent.carrying_shelf is not None
        }
        candidates = [
            shelf.id
            for shelf in self.shelfs
            if shelf.id not in active_shelf_ids and shelf.id not in carried_shelf_ids
        ]
        count = int(
            self.np_random.integers(
                self.batch_size_range[0], self.batch_size_range[1] + 1
            )
        )
        count = min(count, len(candidates))
        if count:
            chosen = self.np_random.choice(candidates, size=count, replace=False)
            self._create_batch([int(shelf_id) for shelf_id in chosen])
```

注意:该函数已具备"排除在途/被携带货物"的候选过滤,切换后无需修改调度本体,只改配置。

### 3.2 优先级字母分配

```218:243:llm_mappo/environment.py
    def _create_batch(self, shelf_ids: Sequence[int]) -> Sequence[Task]:
        if not shelf_ids:
            return ()
        if self.priority_schedule:
            letter = self.priority_schedule[
                self._batch_index % len(self.priority_schedule)
            ]
        else:
            letter = (
                self.initial_priority_label
                if self._batch_index == 0
                else chr(ord("A") + min(self._batch_index, 25))
            )
        batch_id = self._batch_index + 1
        self._batch_index += 1
        tasks = self.task_queue.create_batch(
            shelf_ids, batch_id, letter, self._cur_steps
        )
```

两种模式:
- `priority_schedule` 非空:轮转(适合固定档位实验,如 [A,B,C],但批次>3 会语义回绕)
- `priority_schedule` 为空:首批 `initial_priority_label`,之后按 `chr(ord("A")+index)` 扩展 → **天然符合"越早越高"**,但需把 `initial_priority_label` 从 "B" 改为 "A"

### 3.3 动态归一权重(奖励语义)

```150:155:llm_mappo/rules.py
    def priority_weight(self, label: str) -> float:
        letters = sorted({task.label[0] for task in self.active_tasks})
        if len(letters) <= 1:
            return 1.0
        rank = letters.index(label[0])
        return 0.5 + 1.5 * (len(letters) - rank - 1) / (len(letters) - 1)
```

要点:**权重不是固定档位**,而是按当前活跃任务中的字母集合归一化:
- 仅 1 档字母 → 权重 1.0
- 2 档(A/B)→ A=2.0, B=0.5
- 3 档(A/B/C)→ A=2.0, B=1.25, C=0.5
- 动态扩展(D/E/F)→ 自动归一,无需改代码

奖励 `reward = 5.0 * priority_weight`,因此"最高优先级完成奖励"随环境活跃字母数动态浮动(2.5~10.0),符合 §8.4"搬运优先级越高的货物获得越高的完成奖励"。

### 3.4 蒸馏标签(需适配)

```111:123:llm_mappo/phase3_training.py
def _engagement_targets(env: Phase2Warehouse) -> np.ndarray:
    values = []
    for agent in env.env.agents:
        task = env.env.task_queue.task_for_agent(agent.id)
        if agent.dead or agent.picking_lock_steps or task is None:
            values.append(0.1)
        elif task.label.startswith("A"):
            values.append(0.8)
        elif task.label.startswith("B"):
            values.append(0.5)
        else:
            values.append(0.3)
    return np.asarray(values, dtype=np.float32)
```

动态字母表下,"else → 0.3"会把 C/D/E… 混为一档,无法表达"第 3 批 vs 第 5 批"的优先级差。

---

## 4. 落地方案(分步,按依赖排序)

### Step 1:训练侧参数透传(核心代码改动,需授权)

目标:让 `batch_interval`/`batch_size_range`/`initial_priority_label` 从 config 进入 `Phase2Warehouse`,替代写死的 `max_steps+1`/`(1,1)`/`"B"`。

- 改动点: `llm_mappo/phase2.py` `__post_init__` 的 `make_options`
- 方式: 从 config 读取,带默认值(默认保持当前特例,保证旧实验可复现)
- 校验: `batch_interval >= 2`;`0 < batch_size_range[0] <= batch_size_range[1]`;`initial_priority_label in {"A","B","C"}`

### Step 2:优先级字母模式切换(已确认:从 A 递增,不回绕)

- `priority_schedule`: 保持 `None`(启用动态扩展),而非 `["A","B","C"]`
- `initial_priority_label`: `"A"`(满足"最早批次 = A")
- 这样批次按 `A, B, C, D, E…` 递增,严格对应"越早入库优先级越高";不回绕(决策点 2)

### Step 3:蒸馏标签适配(核心代码改动,需授权;决策点 3 已确认:线性映射)

将 `_engagement_targets` 从"三档查表"改为"按字母表位置线性映射":

```
value = 0.1 + 0.7 * (1 - (rank + 1) / n_active_letters)
```

- `rank`: 该任务字母在**当前活跃字母排序**中的位置(0 = 最高)
- `n_active_letters`: 当前活跃字母数
- 效果: 最高档 ≈ 0.8,最低档 ≈ 0.1,中间按序递减 → 与动态字母表自然对齐,与 `priority_weight` 归一逻辑同构
- 同时更新 `_engagement_label`(诊断采样)与其保持一致

### Step 4:新增 Phase 3 动态入库配置

新建 `configs/phase3_dynamic_ingress.yaml`,基准建议(可调):

```yaml
environment:
  max_steps: 1000
  batch_interval: 100        # 每 100 步入一次货(决策点 1,已确认)
  batch_size_range: [1, 3]   # 每次 1~3 件(决策点 1,已确认)
  initial_priority_label: A
  priority_schedule: []      # 空 → 动态字母扩展
  request_queue_size: 4      # 队列上限略大于 AGV 数,制造选择压力
```

### Step 5:特训场景注入(推迟至 Phase 4,Phase 3 不实施)

> 变更说明:原决策点 4 曾确认"纳入本次范围",现按用户指示调整为 **Phase 3 阶段不做特训**,与动态入库解耦、推迟至 Phase 4 与 LLM 标签一并实施。

在 `reset()` 时按概率注入特训开局(约 25%),模板库覆盖:
- 窄通道双 AGV 对向(一载一空)
- 交叉口高/低优先级相遇
- 电量 15–20% + 载任务 + 充电站邻近

其余 75% 保持自然动态分布,防止分布崩塌。

**为什么推迟**:Phase 3 的蒸馏标签是规则硬编码三档(A/B/C=0.8/0.5/0.3),特训场景若过早注入,学到的行为锚定的是规则标签语义;等 Phase 4a LLM 标签(连续值、语义化)就绪后注入,特训与蒸馏目标同源,效果更好、少一次重训。

### Step 6:行为级评估

新增按行为分组的评估指标(使用未参与训练的种子):
- 窄通道让行率、交叉口高优先级抢行率、低电量充电转向率
- 每个行为组内的完成率/碰撞/死锁单独统计

---

## 5. 验证路径与 Go/No-Go

```
[Phase 3 范围(本方案)]
Step 1-2 落地
  → Step 3 蒸馏适配
  → 先跑 A* 教师 + 规则标签基线(动态环境可解性验证)
       ├─ 基线达到 完成率≥0.95 / 碰撞≤2.0 / 死锁≤0.05 → 继续
       └─ 基线不达标 → 先调 batch_interval / 队列压力
  → Phase 3 动态入库训练(规则标签蒸馏)
  → 行为级评估门 + 整体指标对比

[Phase 4 范围(后续衔接)]
  → Phase 4a LLM 蒸馏(用 §8.4 动态分布生成 e_LLM 标签)
  → Phase 4b 特训场景注入(与 LLM 标签同源,见 Step 5)
```

**注意**:切换动态入库会改变训练状态分布,Phase 3 阶段的 3a/3b 指标数字**不可直接外推**,需在动态分布下重新测量。

---

## 6. 风险与回退

| 风险 | 缓解 |
|---|---|
| 任务持续到达导致队列/目标频繁切换,训练不稳定 | 先按已确认初值 `batch_interval=100` 跑,若不稳再调大至 200 并 warm start 现有 checkpoint |
| 动态字母表数量增长 → 观测/蒸馏维度语义变稀 | 与决策点 2"不回绕"保持一致:限制最大活跃字母数(如 5),超出后**暂停新批入库**待字母消耗,配合 `priority_weight` 归一 |
| 特训注入破坏自然分布(Phase 4 实施) | 保持 75/25 比例,行为级评估监控普通场景指标 |
| 回退 | 保留旧 config(当前特例),Step 1 参数带默认值即可完整复现旧实验 |

---

## 7. 决策点清单(已确认)

| # | 决策点 | 结论 | 落实位置 |
|---|---|---|---|
| 1 | `batch_interval` / `batch_size_range` 初值 | **`100` / `[1, 3]`** | Step 1 参数透传、Step 4 配置 |
| 2 | 优先级字母严格从 A 递增扩展 | **是**,`priority_schedule=None` + `initial_priority_label="A"`,不回绕;批次总数通过 `batch_interval`/`max_steps` 天然受限 | Step 2 |
| 3 | 蒸馏标签三档 → 线性映射 | **改**,`0.1 + 0.7 * (1 - (rank+1)/n)` | Step 3 |
| 4 | 特训场景注入纳入本次范围 | **Phase 3 不做**,推迟至 Phase 4 与 LLM 标签一并实施(2026-08-11 调整) | Step 5(Phase 4) |
| 5 | 切换前先跑 A* 教师基线 | **做**,作为动态环境可解性证明,达标后进入 LLM 阶段 | 第 5 节验证路径 |

> 决策点 2 的补充说明:采用"严格递增、不回绕"后,`max_steps=1000`、`batch_interval=100` 下理论上最多 10 批(字母至 J)。若实测活跃字母数过多导致语义变稀,后续再考虑"限制最大活跃字母数"的限幅策略(见第 6 节风险表)。

---

## 8. 结论

- **环境层已具备动态入库的全部能力**(Phase 1 验证过),本次落地主要是"重新打开 + 参数化 + 蒸馏适配",改动面可控
- 当前训练的机制结论(双头架构、蒸馏配方、奖励结构)**继续有效**,但性能指标需在动态分布下重测
- 建议将当前特例配置固化为回归基线,动态入库作为 **Phase 3** 的必改项
