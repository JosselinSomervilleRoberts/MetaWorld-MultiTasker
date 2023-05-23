## Installation

### 1. Create a conda environment and install pytorch and tensorflow GPU
```bash
conda create -n rlgarage python==3.8
conda init bash
conda activate rlgarage
pip install tensorflow
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

```bash
conda install tensorflow-estimator
```

Also install the Toolbox:
```bash
pip install git+https://github.com/JosselinSomervilleRoberts/JossPythonToolbox.git
```

### 2. Install MuJoco
```
cd /home/ubuntu/
mkdir .mujoco
cd .mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvf mujoco210-linux-x86_64.tar.gz
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
sudo apt-get install patchelf
sudo apt-get install libglew-dev
pip install mujoco
pip install scipy
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

### 3. Install garage
The full instructions are available on the [garage documentation](https://garage.readthedocs.io/en/latest/user/installation.html)
1. Start with `pip install garage`
2. Navigate to the root directory of the garage repository.
3. Then run their setup script to provide the path to your MuJoCo key file:
    - On Linux: `./scripts/setup_linux.sh --mjkey path-to-your-mjkey.txt`
    - On MacOS: `./scripts/setup_macos.sh --mjkey path-to-your-mjkey.txt`
4. Finally `pip install 'garage[mujoco,dm_control]'`

### 4. Install metaworld
```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
pip install gym[box2d]
```