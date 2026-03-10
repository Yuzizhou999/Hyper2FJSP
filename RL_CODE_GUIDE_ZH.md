# RL Code Guide (Chinese)

本文档只覆盖本项目的强化学习部分，方便快速阅读和定位代码。

## 1. 你应该先看哪些文件

按这个顺序阅读最省时间：

1. `params.py`
2. `mo_train.py` (多目标训练入口)
3. `train.py` (训练框架基类)
4. `model/PPO.py`
5. `model/main_model.py`
6. `model/sub_layers.py`
7. `mo_fjsp_env_same_op_nums.py` / `mo_fjsp_env_various_op_nums.py`
8. `mo_test_trained_model.py`

如果只看单目标版本，把 `mo_*` 文件替换成 `fjsp_env_*` 和 `train.py` 即可。

## 2. RL 主流程 (训练)

训练主路径：

- `params.py` 解析配置，构建 `configs`
- `mo_train.py -> MOTrainer` 初始化环境、模型、数据
- `train.py -> Trainer.train()` 执行主循环
- 每个 episode:
  - 环境 `reset/set_initial_data`
  - 策略网络前向得到动作分布 `pi` 与价值 `v`
  - 采样动作并调用环境 `step`
  - 将轨迹写入 `Memory`
- episode 结束后调用 `PPO.update()`
- 周期性调用 validation，按 HV 指标保存最优模型

## 3. 训练入口与控制层

### `mo_train.py`

核心类：`MOTrainer(train.Trainer)`

关键职责：

- `_set_objective_fn`: 将字符串目标转为 `ObjectiveFn` 列表
- `_set_environments`: 构建训练/验证环境，加载验证集与 reference points
- `_initialize_episode`:
  - 重采样训练实例或 `env.reset()`
  - 采样每个并行环境的 preference weights
  - 若模型是 hyper 结构，调用 `policy.assign_preferences(...)`
- `_forward_pass`: 调用 `ppo.policy_old(...)` 得到 `pi, v`
- `_process_reward`: 多目标奖励按 preference 加权求和
- `_call_update_ppo`: 调用 `ppo.update(...)`
- `validate_envs_with_same_op_nums` / `validate_envs_with_various_op_nums`:
  - 生成验证偏好点 (2 目标线性，3/4 目标 Das-Dennis)
  - 计算 Pareto 与归一化 HV

### `train.py`

核心类：`Trainer`

关键职责：

- 管理训练目录、seed、数据路径、训练周期
- 根据数据源选择环境类型 (`FJSPEnvForSameOpNums` 或 `FJSPEnvForVariousOpNums`)
- `train()` 中实现 PPO 训练主循环
- 按 `validate_timestep` 做验证并保存最好 checkpoint

`MOTrainer` 通过覆盖若干 hook（如 `_set_objective_fn`, `_forward_pass`）复用该训练循环。

## 4. PPO 算法实现

### `model/PPO.py`

#### `Memory`

保存一个 rollout 的序列张量：

- 状态：`fea_j_seq`, `op_mask_seq`, `fea_m_seq`, `mch_mask_seq`, `dynamic_pair_mask_seq`, `comp_idx_seq`, `candidate_seq`, `fea_pairs_seq`
- 交互：`action_seq`, `reward_seq`, `val_seq`, `done_seq`, `log_probs`

关键函数：

- `push(state)`: 保存时刻状态
- `transpose_data()`: 从 `[T, B, ...]` 变为扁平 batch 供 PPO 小批次更新
- `get_gae_advantages(...)`: 计算 GAE
  - 单目标：标量优势
  - 多目标：按 `objective_weights` 形成加权优势，并维护多维 value target

#### `PPO`

关键逻辑：

- 根据 `model_architecture` 选择网络
  - 单目标：`DANIEL`
  - 多目标：`MO_DANIEL_ENC_WEIGHT_INPUT` / `MODANIELConditionalOpAndMch` / `MODANIELConditionalOpAndMchFea_Input` / `HYPER_DANIEL`
- `update(...)`:
  - 取 `memory` 扁平数据
  - 计算 advantage 和 value target
  - 多个 `k_epochs` + mini-batch 做 PPO clip 更新
  - loss = `vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss`
  - 末尾 soft-update `policy_old`

## 5. 模型结构

### `model/main_model.py`

基础结构：

- `DualAttentionNetwork`: 工序与机器双注意力编码
- `DANIEL`: 单目标 actor-critic

多目标结构：

- `MO_DANIEL_ENC_WEIGHT_INPUT`
  - 把 preference 直接拼接到 `fea_j` / `fea_m` 输入
- `MODualAttentionNetworkOpAndMch`
  - 使用 MO 注意力块 (`MOMultiHeadOpAttnBlock`, `MOMultiHeadMchAttnBlock`)
- `MODANIELConditionalOpAndMch`
  - 先线性映射 preference，再作为条件输入注意力
- `MODANIELConditionalOpAndMchFea_Input`
  - 在特征输入层显式扩展 preference 维度
- `HYPER_DANIEL`
  - 通过 `HyperActor` 由 preference 生成 actor 参数
  - 支持 `assign_preferences(...)` 先生成参数，减少重复计算

## 6. 子层文件详细说明 (你当前在看的文件)

### `model/sub_layers.py`

#### `MLP`

通用 MLP 封装。`num_layers=1` 时退化为线性层。

#### `Actor`

策略头。输入候选动作特征，输出每个候选的打分。

- 前向：`forward(x, prefences=None)`
- 当 `prefences` 不为空时，可对输出做按偏好加权聚合

#### `Critic`

价值头。输入全局特征，输出状态价值。

- 单目标常为 `[B, 1]`
- 多目标可为 `[B, num_objectives]`（取决于配置）

#### `HyperActor`

超网络 actor，核心思想是“用偏好向量生成 actor 参数”。

关键函数：

- `assign(pref)`:
  - 输入单个或批量 preference
  - 生成 3 层 MLP 的权重和偏置参数
- `forward(x, preferences=None, preference_indices=None)`:
  - 使用已生成参数做 batched 前向
  - `preference_indices` 支持用少量 unique preferences 映射大 batch

这就是 HYPER 模型在多偏好场景下高效推理的关键。

## 7. 环境层 (MDP)

### `fjsp_env_same_op_nums.py`

核心对象：

- `EnvState`: 将 numpy 状态转为 torch tensor，供模型输入
- `FJSPEnvForSameOpNums`: 固定操作数环境

主要状态张量：

- `fea_j_tensor`: 工序特征
- `op_mask_tensor`: 工序邻接/可见性 mask
- `fea_m_tensor`: 机器特征
- `mch_mask_tensor`: 机器关系 mask
- `dynamic_pair_mask_tensor`: 非法 job-machine 对 mask
- `comp_idx_tensor`: 机器竞争关系
- `candidate_tensor`: 当前可调度工序索引
- `fea_pairs_tensor`: job-machine 对特征

### `mo_fjsp_env_same_op_nums.py` / `mo_fjsp_env_various_op_nums.py`

相对单目标环境的扩展点：

- 支持多目标列表 `objective_fns`
- 支持 `use_simple_reward`（训练时更直接的实际值奖励）
- 支持 `use_lb_features`（是否使用下界特征）
- `compute_reward()` 可返回 `[num_envs, num_objectives]`
- 维护 `objective_lower_bounds`，用于 HV 归一化

`various_op_nums` 版本用于每个 job 操作数不一致的数据。

## 8. 测试与评估

### `mo_test_trained_model.py`

主要函数：

- `test_greedy_strategy(...)`
- `test_sampling_strategy(...)`

关键流程：

- 加载训练好的 `ppo.policy`
- 生成偏好集合
- rollout 得到解
- 非支配筛选 (`find_pareto_efficient_solutions`)
- 用 `hvwfg.wfg(...)` 计算归一化 HV
- 可保存每个实例 Pareto 集到 `./pareto_sets/...`

## 9. 配置文件怎么影响 RL

### `params.py`

最关键参数：

- PPO: `lr`, `gamma`, `gae_lambda`, `eps_clip`, `k_epochs`, `minibatch_size`
- 训练规模: `num_envs`, `max_updates`, `reset_env_timestep`, `validate_timestep`
- 模型: `model_architecture`, `hidden_dim_actor`, `hidden_dim_critic`, `use_gamma_beta`
- 多目标: `objective_fn`, `num_preferences_validation`, `num_preferences_test`, `single_value_critic`
- 环境特征: `use_lb_features`, `use_simple_reward`, `deadline_alpha`

另外，`fea_j_input_dim` 会根据 objective 和 `use_lb_features` 动态调整。

## 10. 常用工具函数

### `common_utils.py`

和 RL 最相关的函数：

- `sample_action(p)`
- `eval_actions(p, actions)`
- `greedy_select_action(p)`
- `find_pareto_efficient_solutions(costs)`
- `das_dennis(...)` (用于多目标偏好点生成)

## 11. 一句话理解每个 RL 文件

- `mo_train.py`: 多目标训练总控
- `train.py`: 训练循环基类
- `model/PPO.py`: PPO + GAE 实现
- `model/main_model.py`: DANIEL 及 MO/HYPER 网络定义
- `model/sub_layers.py`: Actor/Critic/HyperActor 基础模块
- `fjsp_env_same_op_nums.py`: 单目标基础环境
- `fjsp_env_various_op_nums.py`: 单目标可变操作数环境
- `mo_fjsp_env_same_op_nums.py`: 多目标固定操作数环境
- `mo_fjsp_env_various_op_nums.py`: 多目标可变操作数环境
- `mo_test_trained_model.py`: 多目标评估与 HV 计算
- `params.py`: 全局超参数入口
- `common_utils.py`: 采样、评估、Pareto 工具

## 12. 你现在可以怎么用这份文档

- 要改模型结构：先看第 5、6 节
- 要改 reward：看第 7 节的 MO 环境
- 要改 PPO 超参或训练节奏：看第 4、9 节
- 要查测试结果怎么来的：看第 8 节

---

如果你愿意，我可以在这份文档基础上再给你补一份“调用关系图版本”（函数级调用链，按 `mo_train.py` 一路点到 `env.step` 与 `ppo.update`）。