# 海洋机器人世界模型

> **Scaffold-first Architecture for World-State Transition Modeling**

本项目面向**海洋机器人世界模型（World Model）**，研究对象固定为：

\[
\text{World State} \rightarrow \text{State Transition} \rightarrow \text{Future World State}
\]

当前第一阶段（Phase 1）只激活 `environment.fluid_state`，使用仿真流体数据建立并验证**环境状态转移预测**。机器人状态、动作条件、多模态观测、概率动力学、流固耦合与 SEAgent 规划集成均作为后续阶段扩展。

> 当前文档基线：**Architecture / Contract v1.8**  
> 当前实现状态：**架构与契约已规划，真实数据链与 SWVT 仍需按 Gate 验证**

---

## 1. 项目目标

项目不是围绕某一种网络或某一种输入模态构建，而是建立一个可以持续替换：

- 数据集；
- 世界状态表示；
- 状态转移模型；
- 解码器；
- 预测目标；
- 评测协议；

的可扩展世界模型脚手架。

核心原则：

> **冻结语义和接口，不冻结算法。**

Phase 1 的核心问题是：

\[
\hat{W}_{t+\tau}
=
F_\theta(W_{\le t}, C, \tau)
\]

在当前阶段：

\[
W_t \approx E_t = \text{environment.fluid_state}
\]

因此实际任务可简化为：

\[
\hat{q}_{t+\tau}
=
F_\theta(q_{\text{history}}, C, \tau)
\]

其中：

- \(q_t\)：当前流体环境状态；
- \(C\)：物理参数、空间域、边界条件、外部 forcing 等上下文；
- \(\tau\)：预测 lead time；
- \(\hat{q}_{t+\tau}\)：未来预测状态。

---

## 2. 当前阶段范围

| 能力 | Phase 1 | 状态 |
|---|---:|---|
| `environment.fluid_state` | IN SCOPE | Planned |
| SWVT | IN SCOPE | Existing asset，待统一接口适配与验证 |
| Deterministic dynamics | IN SCOPE | Planned |
| Lead-time prediction | IN SCOPE | Planned |
| Free rollout | IN SCOPE | Planned |
| Persistence baseline | IN SCOPE | Planned |
| Robot state | OUT | Interface reserved |
| Action conditioning | OUT | Interface reserved |
| Video / sonar fusion | OUT | Interface reserved |
| DiT / probabilistic dynamics | OUT | Probability Gate 通过后再实验 |
| FSI / two-way coupling | OUT | Not in Phase 1 |
| SEAgent planning integration | OUT | Future integration |

Phase 1 **不等于完整机器人世界模型已经实现**。当前阶段只验证世界模型中的**环境动力学分支**。

---

## 3. 总体开发架构

![World Model Architecture](docs/assets/world_model_architecture_v18.png)

系统级稳定入口：

```text
PredictionRequest
        |
        v
WorldModelBackend
        |
        v
WorldPrediction
```

内部允许两类 Backend。

### 3.1 LatentWorldModel

适用于 SWVT、AROMA-inspired encoder、latent DiT 等：

```text
WorldState History
        |
        v
Tensorizer
        |
        v
StateEncoder
        |
        v
Z_history
        |
        v
DynamicsModel
        |
        v
Z_future
        |
        v
StateDecoder
        |
        v
WorldPrediction
```

### 3.2 DirectTransitionWorldModel

适用于 FNO、CNO、Flowers 以及部分 PDE-Transformer 配置：

```text
PredictionRequest
        |
        v
Direct State Transition Backend
        |
        v
WorldPrediction
```

这些模型**不要求为了形式统一被伪拆成 Encoder / Dynamics / Decoder**。

---

## 4. Phase 1 最小状态转移闭环

![Phase 1 Transition Flow](docs/assets/phase1_transition_flow_v18.png)

Phase 1 目标：

```text
The Well / shear_flow
        |
        v
DatasetManifest + TheWellAdapter
        |
        v
TransitionSample
        |
        v
Tensorizer
        |
        v
SWVT / Deterministic Dynamics
        |
        v
WorldPrediction
        |
        +----> Persistence baseline
        |
        v
Evaluator
        |
        v
One-step / Lead-time / Free-rollout / Transition-consistency
```

正式训练前必须先验证数据、张量、契约、模型能力和周期边界。

---

## 5. 核心设计原则

1. **World State 先于 Tensor Shape**  
   先定义状态语义，再定义 `[B, L, C, N_y, N_x]`。

2. **Observation ≠ State**  
   Phase 1 的完整仿真场可以直接构成 `WorldState`；真实视频、声呐、ADCP 等属于 `ObservationFrame`，后续必须经过 `StateEstimator`。

3. **Prediction ≠ Decision**  
   World Model 预测未来状态；任务判断、约束检查和方案选择由 SEAgent / TaskEvaluator / Planner 完成。

4. **Prediction ≠ Fact**  
   `predicted` 状态即使到达其有效时间，也不得自动变成 `observed` 或 `estimated` 事实状态。

5. **时间是一等条件**  
   状态时间使用 `WorldState.timestamp`；模型显式处理历史时间、目标时间与 lead time。

6. **物理上下文与数据 provenance 分离**  
   `WorldContext` 只表达影响状态演化的物理条件；dataset revision、split、license、checksum 属于 `DatasetManifest`。

7. **Fail Closed**  
   schema、模型、边界条件、target time 或数据不兼容时显式失败，不允许 silent dummy fallback。

8. **任务语义与模型能力分离**  
   `TransitionTaskSpec` 定义“要预测什么”，`ModelCapabilities` 定义“模型能做什么”。

9. **概率输出必须经过 Probability Gate**  
   能采样多个未来，不等于得到可信概率分布。

10. **配置驱动、可复现**  
    dataset revision、TaskSpec、BenchmarkProtocol、seed、checkpoint、code commit、normalizer 必须可追溯。

---

## 6. 稳定领域契约

### 6.1 `WorldState`

```yaml
WorldState:
  contract_type: WorldState
  schema_version: ...
  state_id: ...
  state_kind: simulated_ground_truth | observed | estimated | predicted
  timestamp: ...
  world_frame: ...
  components:
    environment:
      fluid_state: ...
    robot: null          # later
    objects: null        # later
    relations: null      # later
  source: ...
  quality: ...
  lineage:
    source_state_ids: []
    source_observation_ids: []
    parent_prediction_id: null
  provenance: ...
```

Phase 1 只激活：

```text
WorldState.components.environment.fluid_state
```

---

### 6.2 `WorldContext`

```yaml
WorldContext:
  contract_type: WorldContext
  schema_version: ...
  context_id: ...
  static_conditions:
    spatial_domains: ...
    geometry: ...
    physics_parameters: ...
    static_boundary_conditions: ...
  temporal_conditions:
    <condition_name>: ConditionSeries
  scenario_id: ...
```

`WorldContext` **不保存**：

- dataset revision；
- split；
- state timestamp；
- requested target time；
- sampling seed；
- num trajectories。

---

### 6.3 `ConditionSeries`

```yaml
ConditionSeries:
  condition_spec_ref: ...
  timestamps: [...]
  values: ...
  source: ...
  interpolation_policy: exact | linear | step | model-specific
  extrapolation_policy: forbidden
  availability_policy: ...
  validity_interval: ...
```

Phase 1 默认禁止超 coverage 的隐式 extrapolation。

---

### 6.4 `ActionSequence`

```yaml
ActionSequence:
  contract_type: ActionSequence
  schema_version: ...
  action_sequence_id: ...
  robot_id: ...
  timestamps_or_intervals: ...
  action_type: ...
  controls_or_commands: ...
  interpolation_or_hold_policy: ...
  validity: ...
```

Phase 1：

```text
actions = None
```

---

### 6.5 `TransitionSample`

训练数据的逻辑单元：

```yaml
TransitionSample:
  contract_type: TransitionSample
  schema_version: ...
  sample_id: ...
  task_spec_ref: ...
  state_history: WorldState[L]
  context: WorldContext
  actions: null
  target_states: WorldState[H]
  trajectory_id: ...
  dataset_manifest_ref: ...
  provenance: ...
```

历史和目标时间从各 `WorldState.timestamp` 派生，不重复保存。

---

### 6.6 `TransitionTaskSpec`

冻结**研究问题**，而不是算法：

```yaml
TransitionTaskSpec:
  contract_type: TransitionTaskSpec
  schema_version: ...
  task_spec_id: ...
  input_components: [...]
  target_components: [...]
  history_selection: ...
  target_time_policy: ...
  required_context_keys: [...]
  action_policy: none | optional | required
  rollout_policy: one-step | lead-time | direct-multi-step | autoregressive
  state_kind_policy: ...
```

它回答：

> **当前 benchmark 到底要求预测什么状态转移？**

它不拥有：

- dataset split；
- metric suite；
- compute budget；
- 模型实现。

---

### 6.7 `PredictionRequest`

```yaml
PredictionRequest:
  contract_type: PredictionRequest
  schema_version: ...
  request_id: ...

  state_history_ref_or_inline: ...
  context_ref_or_inline: ...
  task_spec_ref_or_inline: ...

  target_times: [...]
  actions: null

  prediction_options:
    output_mode: deterministic | ensemble
    num_trajectories: 1
    sampling_seed: null

  strict: true
```

`state_history_ref/state_history`、`context_ref/context`、`task_spec_ref/task_spec` 均要求 **exactly-one-of**。

---

### 6.8 `WorldPrediction`

```yaml
WorldPrediction:
  contract_type: WorldPrediction
  schema_version: ...
  prediction_id: ...
  request_id: ...
  generated_at: ...
  base_state_ids: [...]
  predicted_components: [...]
  trajectories:
    - trajectory_id: ...
      states: WorldState[H]
      sample_weight: null
      rollout_metadata: ...
  uncertainty_summary: ...
  validity: ...
  model_manifest_ref: ...
  context_ref: ...
  task_spec_ref: ...
  status: ...
  warnings: [...]
```

确定性 Phase 1：

```text
K = 1
```

概率模型后续可返回：

```text
K > 1
```

但 raw samples 与 calibration metadata 必须分离。

---

## 7. Fluid State 与空间语义

```yaml
FluidState:
  fields:
    <canonical_field_name>: FieldRef
  spatial_ref: ...
  mask_ref: null
```

Canonical field 示例：

```text
velocity.x
velocity.y
pressure
tracer
```

系统层**不固定为四通道**。

数据集源字段通过：

```text
DatasetManifest.field_mapping
```

映射到 canonical names。

模型的具体通道顺序由：

```text
ModelManifest + TensorSpec + Tensorizer
```

决定。

---

## 8. Manifest 与模型兼容性

### `DatasetManifest`

```text
dataset_id
source / revision / license
file_list / checksums
field_mapping
split_trajectory_ids
time_grid / spatial_metadata
training_statistics_ref
```

### `ModelCapabilities`

```text
input_components / output_components
supported_state_kinds
action_conditioning
output_modes
supported_rollout_modes
history_policy
target_time_policy
supported_grid_types
periodic_axes_policy
padding_policy
supported_spatial_domain
```

### `ModelManifest`

```text
model_id / version / backend_type
compatible_schema_versions
capabilities
required_context
tensor_spec_ref
validated_domain_ref
uncertainty_mode
checkpoint_hash
normalizer_ref
code_commit
```

推理前必须完成：

```text
TransitionTaskSpec
        |
        v
ModelCapabilities compatibility check
        |
        +-- compatible --> Tensorizer / model forward
        |
        +-- incompatible --> TASK_SPEC_INCOMPATIBLE
```

---

## 9. Tensorizer：领域状态与模型张量的防火墙

二维规则网格内部 layout 可以采用：

```text
history: [B, L, C_in, N_y, N_x]
target : [B, H, C_out, N_y, N_x]
```

其中：

- `B`：batch size；
- `L`：历史状态数；
- `H`：未来目标状态数；
- `C_in / C_out`：由 TensorSpec 显式声明；
- `N_y / N_x`：空间网格。

关键边界：

```text
WorldState != Tensor
```

`Tensorizer` 必须负责：

- field mapping；
- channel ordering；
- units；
- dtype；
- layout；
- time/grid validation；
- normalizer；
- padding / periodic boundary compatibility。

模型内部不得根据 shape 自行猜测物理语义。

---

## 10. Phase 1 模型策略

### 主候选：SWVT

当前 SWVT 是：

> **Existing research asset / Primary candidate**

不是永久接口。

Phase 1 计划路径：

```text
Fluid-State History
        |
        v
SWVT StateEncoder
        |
        v
Z_history
        |
        v
Deterministic Dynamics
        |
        v
Z_future
        |
        v
Fluid-State Decoder
        |
        v
PredictedTrajectory(K=1)
```

### 学习型基线

统一通过 `WorldModelBackend` 接入：

| 模型 | 接入方式 | Phase 1 |
|---|---|---|
| SWVT | LatentWorldModel | 主研究候选 |
| PDE-Transformer | Backend / Dynamics backbone | 强 Transformer baseline |
| FNO / CNO | DirectTransitionWorldModel | 非 Transformer baseline |
| Flowers | DirectTransitionWorldModel | 第二轮挑战者 |
| AROMA-like | LatentWorldModel | 后续稀疏/不规则网格 |
| Latent DiT | DynamicsModel | Probability Gate 后 |
| Residual DiT | DynamicsRefiner | Probability Gate 后 |
| DYffusion / Flow Matching | Generative Dynamics | Probability Gate 后 |

---

## 11. Probability Gate

Phase 1 默认使用确定性状态转移。

DiT / flow matching **不会因为模型流行而自动进入主线**。

只有明确存在以下来源之一时，才允许将 probabilistic dynamics 升级为正式研究方向：

```text
partial observation
unknown / stochastic forcing
stochastic boundary
hidden variables
unresolved scales
observation noise
ensemble futures
```

如果给定完整状态和全部已知条件后，动力学基本单值，则概率模型只能作为研究对照。

---

## 12. 数据策略

### 主数据：The Well / `shear_flow`

用途：

- state transition；
- lead-time prediction；
- long rollout；
- state sufficiency；
- condition generalization。

关注信息：

```text
velocity
pressure
tracer
Re / Sc
periodic boundary
continuous trajectories
```

> **禁止在代码中写死分辨率。**

The Well overview、`shear_flow` 专页和 PDE-Transformer 子集文档存在分辨率/规模描述差异。真实开发必须以固定 revision 的 HDF5 metadata、field names、space grid、time grid 和真实 tensor shape 为运行时事实来源。

### 第二验证数据：The Well / `rayleigh_benard`

在主模型稳定后加入，用于验证架构是否只适用于 shear-flow 动力学。

### 后续数据

```text
self-generated CFD / OpenFOAM
robot-environment interaction
real observation
video / sonar / ADCP
```

---

## 13. 数据完整性要求

数据划分顺序：

```text
完整 trajectory
        |
        v
train / valid / test split
        |
        v
生成历史/目标窗口
```

禁止：

```text
先切大量重叠窗口
        |
        v
随机划分窗口到 train/test
```

避免同一底层轨迹的相邻窗口跨 split。

Normalization：

```text
training split statistics only
```

Validation / Test / Inference 必须复用 ModelManifest 指向的同一 normalizer。

---

## 14. 配置驱动

建议目录：

```text
configs/
├── data/
│   ├── shear_flow.yaml
│   └── rayleigh_benard.yaml
├── task/
│   ├── phase1_fluid_transition_smoke.yaml
│   └── phase1_fluid_transition_benchmark.yaml
├── benchmark/
│   └── phase1_shear_flow.yaml
├── model/
│   ├── swvt_det.yaml
│   ├── pde_transformer_mc.yaml
│   ├── fno.yaml
│   └── latent_dit.yaml            # later
├── train/
│   ├── supervised.yaml
│   ├── rollout.yaml
│   └── generative.yaml            # later
└── experiment/
    └── phase1_swvt_shear.yaml
```

Smoke-test 配置中的 `L=4 / H=1` 等值只是**链路检查参数**，不是科学默认值。

---

## 15. 推荐代码结构

```text
world_model/
├── contracts/
│   ├── observation.py
│   ├── world_state.py
│   ├── world_context.py
│   ├── condition.py
│   ├── action.py
│   ├── transition.py
│   ├── task_spec.py
│   ├── prediction.py
│   ├── trajectory.py
│   ├── benchmark.py
│   └── manifests.py
│
├── data/
│   ├── adapters/
│   │   └── the_well/
│   ├── sequence_sampler.py
│   ├── normalization.py
│   └── validation.py
│
├── state_estimation/
│   └── direct_state.py
│
├── models/
│   ├── encoders/
│   │   └── swvt.py
│   ├── dynamics/
│   │   ├── base.py
│   │   ├── deterministic.py
│   │   └── autoregressive.py
│   ├── decoders/
│   │   └── fluid_state.py
│   ├── direct/
│   │   ├── persistence.py
│   │   ├── fno.py
│   │   └── pde_transformer.py
│   └── world_model.py
│
├── tensor/
│   ├── tensor_spec.py
│   └── tensorizer.py
│
├── training/
├── evaluation/
├── inference/
├── registry/
├── configs/
└── tests/
```

后续 DiT、robot encoder、vision/sonar StateEstimator 等只预留 contract / capability / config extension point，**不要提前提交空 implementation**。

---

## 16. 训练流水线

```text
Freeze TransitionTaskSpec + BenchmarkProtocol
        |
        v
Load fixed dataset revision
        |
        v
Build DatasetManifest
        |
        v
Validate fields / time / space / boundary
        |
        v
Split complete trajectories
        |
        v
Build TransitionSample
        |
        v
Compute train-only normalization
        |
        v
Run Persistence baseline
        |
        v
Train deterministic transition
        |
        v
Free rollout
        |
        v
SWVT ablation
        |
        v
Learning-based baseline
```

推荐 SWVT 消融：

```text
shift / no-shift
history length L
state fields
condition input
lead-time / rollout strategy
```

---

## 17. 评测目标

Phase 1 的 PASS 不等于“能输出未来流场图片”。

至少需要评价：

### State Sufficiency

什么变量和多少历史足以支持状态转移？

### Transition Accuracy

单次状态转移是否准确？

### Lead-time / Horizon Curve

误差怎样随预测时间增长？

### Free Rollout

中间不读取真实未来状态时，模型可以稳定推进多久？

### Transition Consistency

例如确定性系统中：

\[
F(q_t, 2\Delta t)
\]

与：

\[
F(F(q_t,\Delta t),\Delta t)
\]

都应与真实：

\[
q_{t+2\Delta t}
\]

比较。

### Physical / Structural Metrics

根据具体物理问题选择：

```text
RMSE / Rel-L2
velocity error
vorticity
spectrum
conservation / physical consistency
```

### Efficiency

```text
training cost
inference latency
peak memory
rollout throughput
```

---

## 18. BenchmarkProtocol

`TransitionTaskSpec` 定义“预测什么”。

`BenchmarkProtocol` 定义“如何公平比较”。

```yaml
BenchmarkProtocol:
  contract_type: BenchmarkProtocol
  schema_version: ...
  benchmark_id: ...
  task_spec_ref: ...
  dataset_manifest_ref: ...
  split_policy_ref: ...
  rollout_horizons: [...]
  metric_suite: [...]
  random_seeds: [...]
  compute_budget_policy: ...
  reporting_policy: ...
```

禁止不同模型自行改变：

- input components；
- target components；
- causal visibility；
- dataset split；
- rollout horizon；
- metric definition；

之后再直接比较结果。

---

## 19. 失败语义

核心状态码建议至少包含：

```text
OK
MODEL_UNAVAILABLE
MODEL_INCOMPATIBLE
TASK_SPEC_INCOMPATIBLE
INVALID_INPUT
SCHEMA_MISMATCH
TIME_RANGE_INVALID
CONDITION_COVERAGE_INVALID
BOUNDARY_POLICY_UNSUPPORTED
OUTPUT_MODE_UNSUPPORTED
INSUFFICIENT_STATE
OUT_OF_VALIDATED_RANGE
PREDICTION_FAILED
UNCERTAINTY_UNAVAILABLE
```

生产路径禁止：

```text
real model failed
        |
        v
dummy / persistence fallback
        |
        v
status = OK
```

Persistence / mock 只能作为显式 benchmark 或测试替身。

---

## 20. 开发 Gate

进入 SWVT 正式训练前必须完成：

| Gate | 验收 |
|---|---|
| **G0 Contract Gate** | schema/version、timestamp SSOT、lineage、exactly-one-of、serialization/migration |
| **G1 Data Gate** | 固定 revision The Well fixture、field mapping、GridSpec、split/provenance |
| **G2 Tensor Gate** | channel/layout/units/grid/normalizer、roundtrip、no-future-leakage |
| **G3 Backend Gate** | ModelManifest compatibility、strict request、fail-closed |
| **G4 Evaluation Gate** | Persistence + unified Evaluator 跑通 |
| **G5 SWVT Boundary Gate** | periodic/window/shift/mask/padding/rectangular-grid 行为有测试 |
| **G6 State-Origin Gate** | predicted state + lineage 正确；预测不会自动 promotion |

---

## 21. 首轮开发任务

建议按独立 commit 推进：

```text
WM-00  Core contracts
   ↓
WM-01  DatasetManifest + TheWellAdapter fixture
   ↓
WM-02  TensorSpec + Tensorizer
   ↓
WM-03  ModelManifest + Backend/Dynamics contracts
   ↓
WM-04  Persistence + Evaluator smoke
   ↓
WM-05  SWVT integration + Boundary Gate
   ↓
WM-06  Free rollout + Transition consistency
   ↓
WM-07  FNO or PDE-Transformer baseline
   ↓
WM-08  Full shear_flow + ablation
   ↓
WM-09  Freeze Phase 1 BenchmarkProtocol
```

原则：

> **WM-00～WM-04 通过后，再把 SWVT 结果作为正式科研实验。**

---

## 22. 后续路线

```text
Phase 0
World Model Scaffold
        ↓
Phase 1
Environment Fluid-State Transition
        ↓
Phase 2
Probability Dynamics
[only after Probability Gate]
        ↓
Phase 3
Environment + Robot + Action
        ↓
Phase 4
Environment + Robot + Objects + Relations
        ↓
Phase 5
Partial Observation / Real Sensor State Estimation
        ↓
SEAgent Task Evaluation / Planning
```

后续双向流固耦合只在：

```text
robot/object motion significantly changes environment
```

且该相互作用属于预测目标时引入。

---

## 23. Phase 1 明确不做

Phase 1 不承担：

```text
真实海洋泛化结论
机器人 action-conditioned transition
双向 FSI
视频/声呐多模态融合
SEAgent 自动任务授权
未经定义不确定性来源的“概率世界模型”
不同论文/数据版本/训练预算的直接排行榜
silent dummy fallback
```

---

## 24. 当前架构决策

| 问题 | 当前决策 |
|---|---|
| 项目研究什么 | 世界模型状态转移预测 |
| Phase 1 状态 | `environment.fluid_state` |
| Phase 1 主数据 | The Well / `shear_flow` |
| 第二验证数据 | The Well / `rayleigh_benard` |
| 当前主候选 | Shifted-Window ViT（SWVT） |
| 是否锁死 SWVT | 否 |
| SWVT 的位置 | LatentWorldModel 的 StateEncoder / Dynamics 路径 |
| DiT 的位置 | DynamicsModel / DynamicsRefiner，受 Probability Gate 控制 |
| FNO / Flowers | DirectTransitionWorldModel |
| 是否需要 FSI | Phase 1 不需要 |
| 顶层稳定 API | `PredictionRequest -> WorldModelBackend -> WorldPrediction` |
| 最先开发 | Contracts + Data + Tensorizer + Backend + Persistence/Evaluator |
| 正式训练前 | 必须通过 G0–G5 |
| 世界模型与 SEAgent | World Model 提供未来状态；SEAgent 负责任务评估与规划 |

---

## 25. 参考资料

1. Hafner et al., **Mastering diverse control tasks through world models**, Nature, 2025.  
   https://www.nature.com/articles/s41586-025-08744-2

2. Zhou et al., **DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning**, ICML 2025.  
   https://proceedings.mlr.press/v267/zhou25t.html

3. Serrano et al., **AROMA: Preserving Spatial Structure for Latent PDE Modeling with Local Neural Fields**, NeurIPS 2024.  
   https://proceedings.neurips.cc/paper_files/paper/2024/hash/185a120a3f709187e68bd092e6098851-Abstract-Conference.html

4. Herde et al., **Poseidon: Efficient Foundation Models for PDEs**, NeurIPS 2024.  
   https://arxiv.org/abs/2405.19101

5. Holzschuh et al., **PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations**, ICML 2025.  
   https://proceedings.mlr.press/v267/holzschuh25a.html

6. Ohana et al., **The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning**, NeurIPS 2024.  
   https://proceedings.neurips.cc/paper_files/paper/2024/file/4f9a5acd91ac76569f2fe291b1f4772b-Paper-Datasets_and_Benchmarks_Track.pdf

7. **The Well API / Data Format / Dataset Tutorial**  
   https://polymathic-ai.org/the_well/

8. Muser et al., **Flowers: A Warp Drive for Neural PDE Solvers**, 2026.  
   https://github.com/t-muser/flowers

9. Liu et al., **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**, ICCV 2021.  
   https://arxiv.org/abs/2103.14030

10. **The Well: Periodic shear flow**  
    https://polymathic-ai.org/the_well/datasets/shear_flow/

11. **PDE-Transformer: The Well preprocessing/subset**  
    https://tum-pbs.github.io/pde-transformer/datasets/the_well.html

12. **The Well repository / benchmark workflow**  
    https://github.com/PolymathicAI/the_well

---

## 26. 当前开发入口

当前不建议直接从“大规模 SWVT 训练”开始。

首先完成：

```text
WorldState
WorldContext
TransitionTaskSpec
PredictionRequest
WorldPrediction
DatasetManifest
ModelManifest
BenchmarkProtocol
```

然后使用固定 revision 的 The Well 小样本跑通：

```text
TheWellAdapter
    -> TransitionSample
    -> Tensorizer
    -> Persistence
    -> WorldPrediction
    -> Evaluator
```

该链路通过后，再进入 SWVT 状态转移训练。

---

## License

项目许可证、第三方数据许可证及模型许可证应分别记录。数据集许可必须以固定 `DatasetManifest.revision` 对应的官方来源为准。
