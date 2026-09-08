# Installation Guide

## System Requirements

- **Operating System**: Recommended Ubuntu 20.04 or later  
- **GPU**: Nvidia GPU  
- **Driver Version**: Recommended version 525 or later  
- **Python Version**: Python 3.8
- **Supported Robots**: Adam Lite

---

## 1. Creating a Virtual Environment

It is recommended to run training or deployment programs in a virtual environment. Conda is recommended for creating virtual environments. If Conda is already installed on your system, you can skip step 1.1.

### 1.1 Download and Install MiniConda

MiniConda is a lightweight distribution of Conda, suitable for creating and managing virtual environments. Use the following commands to download and install:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

After installation, initialize Conda:

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 Create a New Environment

Use the following command to create a virtual environment:

```bash
conda create -n pndbotics-rl python=3.8 -y
```

### 1.3 Activate the Virtual Environment

```bash
conda activate pndbotics-rl
```

---

## 2. Installing Dependencies

### 2.1 Install PyTorch

PyTorch is a neural network computation framework used for model training and inference. Install it using the following command:

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2.2 Install Isaac Gym

Isaac Gym is a rigid body simulation and training framework provided by Nvidia.

#### 2.2.1 Download

Download [Isaac Gym](https://developer.nvidia.com/isaac-gym) from Nvidia’s official website.

#### 2.2.2 Install

After extracting the package, navigate to the `isaacgym/python` folder and install it using the following commands:

```bash
cd isaacgym/python
pip install -e .
```

#### 2.2.3 Verify Installation

Run the following command. If a window opens displaying 1080 balls falling, the installation was successful:

```bash
cd examples
python 1080_balls_of_solitude.py
```

If you encounter any issues, refer to the official documentation at `isaacgym/docs/index.html`.

### 2.3 Install rsl_rl

`rsl_rl` is a library implementing reinforcement learning algorithms.

#### 2.3.1 Download

Clone the repository using Git:

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git
```

#### 2.3.2 Switch Branch

Switch to the v1.0.2 branch:

```bash
cd rsl_rl
git checkout v1.0.2
```

#### 2.3.3 Install

```bash
pip install -e .
```

### 2.4 Install pnd_rl_gym

#### 2.4.1 Download

Clone the repository using Git:

```bash
git clone https://github.com/pndbotics/pnd_rl_gym.git
```

#### 2.4.2 Install

Navigate to the directory and install it:

```bash
cd pnd_rl_gym
pip install -e .
```

**Note**: This step will automatically install the following dependencies:
- `matplotlib`: Data visualization
- `numpy==1.20`: Numerical computing library
- `tensorboard`: Training visualization tool
- `mujoco==3.2.3`: Mujoco simulator
- `pyyaml`: YAML configuration file parser
- `wandb`: Experiment tracking and visualization (optional)

### 2.5 Configure WandB (Optional)

`WandB (Weights & Biases)` is a powerful experiment tracking and visualization tool for monitoring training progress, managing model versions, and comparing different training runs.

#### 2.5.1 Create an Account

Visit [https://wandb.ai/](https://wandb.ai/) to create a free account.

#### 2.5.2 Login

```bash
wandb login
```

You'll be prompted to enter your API key, which can be found at [https://wandb.ai/authorize](https://wandb.ai/authorize).

#### 2.5.3 Verify Installation (Optional)

```bash
python scripts/test_wandb.py
```

#### 2.5.4 Using WandB

Add the `--wandb` flag when training to enable it:

```bash
python legged_gym/scripts/train.py --task=adam_lite --wandb
```

**Note**: If you don't need WandB, you can skip this step. Simply don't add the `--wandb` flag during training.

### 2.6 Install pndbotics_sdk_py (Optional)

`pndbotics_sdk_py` is a library used for communication with real robots. If you need to deploy the trained model on a physical robot, install this library.

#### 2.6.1 Download

Clone the repository using Git:

```bash
git clone https://github.com/pndbotics/pnd_sdk_python.git
```

#### 2.6.2 Install

Navigate to the directory and install it:

```bash
cd pnd_sdk_python
pip install -e .
```

---

## 3. Verify Installation

### 3.1 Verify Isaac Gym

As mentioned earlier, run the Isaac Gym example:

```bash
cd isaacgym/python/examples
python 1080_balls_of_solitude.py
```

### 3.2 Verify pnd_rl_gym

Run a training command to test if the environment is working properly:

```bash
cd pnd_rl_gym
python legged_gym/scripts/train.py --task=adam_lite --headless --num_envs=64 --max_iterations=1
```

If the command executes successfully and generates log files, the installation is complete.

---

## 4. Troubleshooting

### 4.1 Isaac Gym Installation Failure

- Ensure Nvidia drivers are installed (version 525+)
- Check that Python version is 3.8
- Refer to the troubleshooting guide in `isaacgym/docs/index.html`

### 4.2 CUDA-related Errors

- Ensure PyTorch's CUDA version is compatible with your system's CUDA version
- Use `nvidia-smi` to check your system's CUDA version

### 4.3 numpy Version Conflicts

- This project requires `numpy==1.20`
- If you encounter version conflicts, force reinstall with:
  ```bash
  pip install numpy==1.20 --force-reinstall
  ```

---

## Summary

After completing the above steps, you are ready to run the related programs in the virtual environment. The complete workflow is:

1. **Training**: `python legged_gym/scripts/train.py --task=adam_lite`
2. **Play Verification**: `python legged_gym/scripts/play.py --task=adam_lite`
3. **Sim2Sim Deployment**: `python deploy/deploy_mujoco/deploy_mujoco.py adam_lite.yaml`
4. **Sim2Real Deployment**: `python deploy/deploy_real/deploy_real.py {network_interface} adam_lite.yaml`

For detailed usage instructions, please refer to the main [README.md](../README.md) documentation.

If you encounter any issues, refer to the official documentation of each component or check if the dependencies are installed correctly.

