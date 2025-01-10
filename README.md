# Pi_rl_baseline
English| [中文](README_zh.md)

This work provides a reinforcement learning environment based on NVIDIA Isaac Gym. For Gaoqing Mechatronics' bipedal robot Pi, Pi_rl_baseline also integrates the sim2sim framework from Isaac Gym to Mujoco, allowing users to verify the trained strategies in different physical simulations.

## Environment Installation
### System Requirements
- **Operating System**: Ubuntu 18.04 or higher is recommended
- **Graphics Card**: Nvidia Graphics Card
- **Driver Version**: 525 or higher is recommended


### Deployment steps
1. Use Conda to create a virtual environment and run training or deployment programs in the virtual environment. The following commands download and install miniconda
`mkdir -p ~/miniconda3
&& wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
&& bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
&& rm ~/miniconda3/miniconda.sh` Initialize Conda: `~/miniconda3/bin/conda init --all
&& source ~/.bashrc`

2. Generate a new Python virtual environment using Python 3.8 `conda create -n myenv python=3.8`
3. Install PyTorch 1.13 with Cuda-11.7:
- `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
4. Install numpy-1.23 `conda install numpy=1.23`
5. Install Isaac Gym:
- Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
- `cd isaacgym/python && pip install -e .`
- Run examples `cd examples && python 1080_balls_of_solitude.py`
- See `isaacgym/docs/index.html` for troubleshooting.
6. Install Pi_rl_baseline:
- Clone this repository `git clone https://github.com/HighTorque-Robotics/pi_rl_baseline.git`
- `cd Pi_rl_baseline && pip install -e .`



## Usage Guide

#### Examples

```bash
# Launching PPO Policy Training for 'v1' Across 4096 Environments
# This command initiates the PPO algorithm-based training for the humanoid task.
python scripts/train.py --task=pai_ppo --run_name v1 --headless --num_envs 4096

# Evaluating the Trained PPO Policy 'v1'
# This command loads the 'v1' policy for performance assessment in its environment. 
# Additionally, it automatically exports a JIT model, suitable for deployment purposes.
python scripts/play.py --task=pai_ppo --run_name v1

# Implementing Simulation-to-Simulation Model Transformation
# This command facilitates a sim-to-sim transformation using exported 'v1' policy.
python scripts/sim2sim.py --load_model /path/to/logs/Pai_ppo/exported/policies/policy_1.pt

# Run our trained policy
python scripts/sim2sim.py --load_model /path/to/logs/Pai_ppo/exported/policies/policy_example.pt

```

#### 1. Training and deployment
- **Run the following command for training:**
```
python humanoid/scripts/train.py --task=pai_ppo --run_name run_name
```
- **If you want to deploy the trained policy in Gym, execute:**
```
python humanoid/scripts/play.py --task=pai_ppo --run_name run_name
```
- By default, the latest model from the last run is loaded. However, you can select a different run model by adjusting `run_name` and `checkpoint` in the training configuration (`task` specifies the task type for the current training || deployment, and `run_name` is the name of the training run corresponding to the current training || deployment. For example, if there was a training run named test_v1 before and you want to load the model of the 100th checkpoint of that run, you can set run_name test_v1 and checkpoint 100 in the configuration file).

#### 2. Sim-to-sim

- **Use Mujoco to perform sim2sim deployment using the following command:**
```
python scripts/sim2sim.py --load_model /path/to/export/model.pt
```

#### 3. Parameter Description
- **CPU and GPU Usage**: Run simulation using CPU, set `--sim_device=cpu` and `--rl_device=cpu`. Run simulation using specified GPU, set `--sim_device=cuda:{0,1,2...}` and `--rl_device={0,1,2...}`. Note that `CUDA_VISIBLE_DEVICES` does not apply, and matching `--sim_device` and `--rl_device` settings is critical.
- **Headless Operation**: Include `--headless` for renderless operation.
- **Rendering Control**: Press "v" to pause rendering during training.
- **Policy Location**: The trained policy is saved in `humanoid/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`


## Code Structure

1. Each environment depends on an `env` file (`legged_robot.py`) and a `config` file (`legged_robot_config.py`). The latter contains two classes: `LeggedRobotCfg` (containing all environment parameters) and `LeggedRobotCfgPPO` (representing all training parameters).
2. Both `env` and `config` classes use inheritance.
3. A non-zero reward scale specified in `cfg` adds the function of the corresponding name to the total reward.
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. Registration can be inside `envs/__init__.py` or outside this repository.


## Add a new environment 

The base environment "legged_robot" is designed to solve the task of locomotion on rough terrain. However, the corresponding configuration does not specify a robot model (URDF/MJCF) and lacks a reward function. To add a new environment, follow these steps:

1. Create a new environment
- Create a new folder in the `envs/` directory with a configuration file named "<your_env>_config.py".
- Make sure the new configuration inherits from the existing environment configuration to leverage existing features and functions

2. Use a new robot model
- Place the corresponding URDF (MJCF) file in the `resources/` folder.
- In the `cfg/` file, set the path to the model, define the body name, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
- In `train_cfg`, set `experiment_name` and `run_name`.

3. Create your environment in `<your_env>.py'. Inherit an existing environment, overwrite the required functions or add your custom reward function.
4. Register your environment in `humanoid/envs/__init__.py`.
5. Modify or adjust other parameters in `cfg` or `cfg_train` as needed.
- To remove the reward, set its scale to zero to avoid modifying other environment's parameters.
6. If you want a new robot/environment to perform sim2sim, you need to modify `humanoid/scripts/sim2sim.py`:
- Check the robot's joint mapping between URDFs.
- Change the robot's initial joint positions according to the trained policy.

## Acknowledgment

The implementation of livelybot_rl_control relies on resources from [legged_gym](https://github.com/leggedrobotics/legged_gym)  and [humanoid-gym](https://github.com/roboterax/humanoid-gym) projects.

