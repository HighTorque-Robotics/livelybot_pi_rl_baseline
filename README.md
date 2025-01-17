# Pi_rl_baseline
[English](README_EN.md) | 中文

该工作提供了一个基于 NVIDIA Isaac Gym 的强化学习环境，对于高擎机电的双足机器人 Pi， Pi_rl_baseline 还整合了从 Isaac Gym 到 Mujoco 的sim2sim框架，使用户能够在不同的物理模拟中验证训练得到的策略

## 环境安装
### 系统要求
  - **操作系统**：推荐使用 Ubuntu 18.04 或更高版本  
  - **显卡**：Nvidia 显卡  
  - **驱动版本**：建议使用 525 或更高版本  


### 部署步骤
1. 使用 Conda 创建虚拟环境，在虚拟环境中运行训练或部署程序。以下命令下载与安装miniconda
    `mkdir -p ~/miniconda3
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    && rm ~/miniconda3/miniconda.sh` 初始化 Conda：`~/miniconda3/bin/conda init --all
    && source ~/.bashrc`

2. 使用 Python 3.8 生成新的 Python 虚拟环境 `conda create -n myenv python=3.8`
3. 安装带有 Cuda-11.7 的 PyTorch 1.13:
   - `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
4. 安装 numpy-1.23 `conda install numpy=1.23`
5. 安装 Isaac Gym:
   - 从以下网站下载并安装 Isaac Gym Preview 4 https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - 运行示例 `cd examples && python 1080_balls_of_solitude.py`
   - 请参阅 `isaacgym/docs/index.html` 以进行故障排除。
6. 安装 Pi_rl_baseline:
   - 克隆此仓库`git clone https://github.com/HighTorque-Robotics/pi_rl_baseline.git`
   - `cd Pi_rl_baseline && pip install -e .`



## 使用指南

#### 示例

```bash
# 在 4096 个环境中启动“v1”的 PPO 策略训练
# 该命令启动基于 PPO 算法的人形任务训练
python scripts/train.py --task=pai_ppo --run_name v1 --headless --num_envs 4096

# 评估经过训练的 PPO 策略“v1”
# 该命令加载“v1”策略以在其环境中进行性能评估
# 此外，它还会自动导出适合部署目的的 JIT 模型
python scripts/play.py --task=pai_ppo --run_name v1

# 通过使用Mujoco实现sim2sim
python scripts/sim2sim.py --load_model /path/to/logs/Pai_ppo/exported/policies/policy_1.pt

# 运行我们提供的训练好的policy
python scripts/sim2sim.py --load_model /path/to/logs/Pai_ppo/exported/policies/policy_example.pt

```

#### 1. 训练与部署
- **运行以下命令进行训练：**
  ```
  python humanoid/scripts/train.py --task=pai_ppo --run_name run_name
  ```
- **如果想要在 Gym 中部署已训练的 policy，请执行:**
  ```
  python humanoid/scripts/play.py --task=pai_ppo --run_name run_name
  ```
- 默认情况下，会加载上次运行的最新模型。但是，可以通过调整训练配置中的 `run_name` 和 `checkpoint` 来选择其他运行模型（`task`为当前 训练 || 部署 指定任务类型，`run_name`为当前 训练 || 部署 对应的训练运行名称，例如，如果之前有一个名为 test_v1 的训练运行，并且想要加载该运行的第 100 个检查点的模型，可以在配置文件中设置 run_name test_v1 和 checkpoint 100）。

#### 2. Sim-to-sim

- **利用 Mujoco 使用以下命令执行 sim2sim 部署:**
  ```
  python scripts/sim2sim.py --load_model /path/to/export/model.pt
  ```

#### 3. 参数说明
- **CPU and GPU Usage**: 使用CPU运行仿真，同时设置`--sim_device=cpu` 和 `--rl_device=cpu`. 使用指定GPU运行仿真，同时设置  `--sim_device=cuda:{0,1,2...}` 和 `--rl_device={0,1,2...}` .请注意，`CUDA_VISIBLE_DEVICES` 不适用，并且匹配 `--sim_device` 和 `--rl_device` 设置至关重要。
- **Headless Operation**: 包含 `--headless` 以进行无渲染操作。
- **Rendering Control**: 按“v”在训练期间暂停渲染。
- **Policy Location**: 训练后的策略保存在 `humanoid/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`


## 代码结构

1. 每个环境都依赖于一个 `env` 文件（`legged_robot.py`）和一个 `config` 文件（`legged_robot_config.py`）。后者包含两个类：`LeggedRobotCfg`（包含所有环境参数）和 `LeggedRobotCfgPPO`（表示所有训练参数）。
2. `env` 和 `config` 类都使用继承。
3. `cfg` 中指定的非零奖励尺度将相应名称的函数加入到总奖励。
4. 必须使用 `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)` 注册任务。注册可以在 `envs/__init__.py` 内，也可以在此存储库之外。


## 添加新环境

基础环境“legged_robot”旨在解决崎岖地形上的运动任务。然而，相应的配置未指定机器人模型（URDF/MJCF），且缺少奖励函数。若需添加新环境，请遵循以下步骤：

1. 创建新环境
   - 进入`envs/`目录中创建一个新文件夹，其中包含名为“<your_env>_config.py”的配置文件。
   - 确保新配置继承自现有环境配置，以便利用现有功能和特性

2. 使用新机器人模型
   - 将相应URDF（MJCF）文件放入`resources/`文件夹中。
   - 在`cfg/`文件中，设置模型的路径，定义主体名称、default_joint_positions 和 PD 增益。指定所需的`train_cfg`和环境的名称（python 类）。
   - 在`train_cfg`中，设置`experiment_name`和`run_name`。

3. 在“<your_env>.py”中创建您的环境。继承现有环境，覆盖所需函数或添加您的自定义奖励函数。
4. 在 `humanoid/envs/__init__.py` 中注册您的环境。
5. 根据需要修改或调整 `cfg` 或 `cfg_train` 中的其他参数。
   - 要移除奖励，请将其比例设置为零，避免修改其他环境的参数。
6. 如果想要一个新的机器人/环境来执行 sim2sim，需要修改 `humanoid/scripts/sim2sim.py`：
   - 检查 URDF 之间机器人的关节映射。
   - 根据训练的策略更改机器人的初始关节位置。

## 致谢

livelybot_rl_control 的实现依赖于 [legged_gym](https://github.com/leggedrobotics/legged_gym) 和 [humanoid-gym](https://github.com/roboterax/humanoid-gym) 项目的资源。