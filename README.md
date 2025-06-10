# ActiveGAMER

# Active GAussian Mapping through Efficient Rendering (CVPR - 2025)

<a href='https://oppo-us-research.github.io/ActiveGAMER-website/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2501.06897'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=2sfVMuZq92Y)
<a href='https://github.com/oppo-us-research/ActiveGAMER-website/blob/main/static/images/cvpr25_poster_activegamer.png'><img src='assets/poster_badge.png' width=78 height=21></a>


This is an official implementation of the paper "ActiveGAMER: Active GAussian Mapping through Efficient Rendering". 

[__*Liyan Chen*__](https://scholar.google.com/citations?user=ZU9JhNYAAAAJ&hl=en)<sup>\*&dagger;1,2</sup>, 
[__*Huangying Zhan*__](https://huangying-zhan.github.io/)<sup>\*&dagger;&ddagger;1</sup>, 
[Kevin Chen](https://github.com/kuantingchen04)<sup>1</sup>, 
[Xiangyu Xu](https://scholar.google.com/citations?user=Pt4q_QkAAAAJ&hl=en)<sup>1</sup>, 
<br>
[Qingan Yan](https://yanqingan.github.io/)<sup>1</sup>, 
[Changjiang Cai](https://www.changjiangcai.com/)<sup>1</sup>, 
[Yi Xu](https://www.linkedin.com/in/yi-xu-42654823/)<sup>1</sup> 

<sup>1</sup> OPPO US Research Center, 
<sup>2</sup> Stevens Institute of Technology

<sup>*</sup> Co-first authors with equal contribution </br>
<sup>&dagger;</sup> Work done as an intern at OPPO US Research Center </br>
<sup>&ddagger;</sup> Corresponding author </br>


## Table of Contents
- [📁 &nbsp; Repo Structure](#repo-structure)
- [🛠️ &nbsp; Installation](#installation) 
- [💿 &nbsp; Dataset Preparation](#dataset-preparation)
- [🏃‍♂️ &nbsp; Running ActiveGAMER](#running_active_gamer)
- [🔎 &nbsp; Evaluation](#evaluation)
- [📜 &nbsp; License](#license)
- [🤝 &nbsp; Acknowledgement](#acknowledgement)
- [📖 &nbsp; Citation](#citation)

<h2 id="repo-structure"> 📁 Repo Structure  </h2>

```
# Main directory
├── ActiveGAMER (ROOT)
│   ├── assets                                      # README assets
│   ├── configs                                     # experiment configs
│   ├── data                                        # dataset
│   │   └── MP3D                                    # Matterport3D for Habitat data
│   │   └── ReplicaSLAM                             # Replica SLAM Dataset
│   │   └── replica_v1                              # Replica Dataset v1
│   ├── envs                                        # environment installation 
│   ├── results                                     # experiment results
│   ├── scripts                                     # scripts
│   │   └── activegamer                             # running ActiveGAMER scripts
│   │   └── data                                    # data related scripts
│   │   └── evaluation                              # evaluation related scripts
│   │   └── installation                            # installation related scripts
│   ├── src                                         # source code
│   │   └── data                                    # data code
│   │   └── evaluation                              # evaluation code
│   │   └── layers                                  # pytorch layers
│   │   └── main                                    # main framework code
│   │   └── planer                                  # planer code
│   │   └── simulator                               # simulator code
│   │   └── slam                                    # SLAM code
│   │   └── utils                                   # utility code
│   │   └── visualization                           # visualization code
│   ├── third_party                                 # third_party


# Data structure
├── data                                            # dataset dir
│   ├── MP3D                                        # Matterport3D data
│   │   └── v1
│   │       └── tasks
│   │           └── mp3d_habitat
│   │               ├── 1LXtFkjw3qL
│   │               └── ...
│   ├── replica_v1                                  # Replica-Dataset
│   │   └── apartment_0
│   │       └── habitat
│   │           └── replicaSDK_stage.stage_config.json
│   │   └── ...
│   ├── Replica                                     # Replica SLAM Dataset
│   │   └── office_0
│   │   └── ...

# Configuration structure
├── configs                                         # configuration dir
│   ├── default.py                                  # Default overall configuration
│   ├── MP3D                                        # Matterport3D 
│   │   └── ActiveGAMER                             # ActiveGAMER default configuration for MP3D
│   │       └── mp3d_splatam_s.yaml                 # SplaTAM-small default configuration for MP3D
│   │       └── mp3d_splatam.yaml                   # SplaTAM default configuration for MP3D
│   │   └── {SCENE}
│   │       └── {EXP}.py                            # experiment-specific overall configuraiton, inherit from default.py
│   │       └── habitat.py                          # scene-specific HabitatSim configuration
│   │   └── ...
│   ├── Replica                                     # Replica
│   │   └── ...

NOTE: default.py and {EXP}.py have the highest priority that can override configurations in other config files (e.g. *.yaml, habitat.py)

# Result structure
├── results                                         # result dir
│   ├── Replica                                     # Replica result
│   │   └── office_0
│   │       └── {EXPERIMENT_SETUP}
│   │           └── run_{COUNT}
│   │               └── eval_3d                     # 3D Reconstruction evaluation results
│   │                   └── {STAGE}
│   │                       └── eval_3d_result.txt  # result summary
│   │                       └── pt_err_*.ply        # 3D error visualization
│   │                       └── rec_pc_tri.ply      # reconstructed point cloud extracted from 3DGS
│   │               └── logger                      # SplaTAM event logging
│   │               └── splatam
│   │                   └── eval                    # Training set evaluation
│   │                   └── eval_{STAGE}            # Novel View Synthesis evaluation
│   │                       └── plots               # NVS qualitative results
│   │                       └── *.txt               # NVS quantitative results
│   │                   └── {STAGE}                 # 3DGS Parameters
│   │                   └── config.py               # splatam configuration
│   │               └── main_cfg.json               # main configuration
│   ├── MP3D                                        # Matterport3D result
│   │   └── GdvgFV5R1Z5
│   │       └── {EXPERIMENT_SETUP}
│   │           └── run_{COUNT}
│   │               └── ...
│   │   └── ...
│   │   └── ...
```

<h2 id="installation"> 🛠️ Installation </h2>

### Install ActiveGAMER

```
# Clone the repo with the required third parties.
git clone --recursive https://github.com/oppo-us-research/ActiveGAMER.git

# We assume ROOT as the project directory.
cd ActiveGAMER
ROOT=${PWD}
```

In this repo, we provide two types of environement installations: Docker and Anaconda.

Users can optionally install one of them. The installation process includes: 

- installation of [Habitat-Sim](https://github.com/facebookresearch/habitat-sim), where we install our updated [Habitat-Sim](https://github.com/Huangying-Zhan/habitat-sim), where the geometry compilation is updated.  

- installation of [SplaTAM](https://github.com/spla-tam/SplaTAM), which is used as ActiveGAMER's mapping backbone

- installation of other packages required to run ActiveGAMER.

### [Optional 1] Docker Environment

Follow the steps to install the Docker environment: 
```
# Build Docker image
bash scripts/installation/docker_env/build.sh

# Run Docker container. Update the mount volume accordingly.
bash scripts/installation/docker_env/run.sh

# Activate conda env in Docker Env
conda activate activegamer
```

### [Optional 2] Conda Environment

Follow the steps to install the conda environment

```
# Build conda environment
bash scripts/installation/conda_env/build.sh

# Activate conda env
source activate activegamer
```

<h2 id="dataset-preparation"> 💿 Dataset Preparation   </h2>

### Replica Data

Follow the steps to download [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset/tree/main).
```
# Download Replica data and save as data/replica_v1.
# This process can take a while.
bash scripts/data/replica_download.sh data/replica_v1

# Once the donwload is completed, create modified Habitat Simulator configs that adjust the coordinate system direction.
# P.S. we adjust the config so the coordinates matches with the mesh coordinates.
bash scripts/data/replica_update.sh data/replica_v1
```

We use three versions of Replica data for different purposes.

1. ReplicaSLAM: a SLAM dataset with predefined trajectory. Used for passive mapping experiement, and pose initialization for the first frame.
2. ReplicaSLAM-Habitat: Use same trajectory in (1) but using HabitatSim to re-generate the RGBD data, as (1)'s generated RGBD does not match with HabitatSim's simulation data. e.g. different lighting configuration.
3. ReplicaNVS: Novel View Synthesis evaluation data.

```
# (1) Download ReplicaSLAM Data and save as data/Replica
bash scripts/data/replica_slam_download.sh

# (2) Generate (1) using HabitatSim
bash scripts/data/generate_replica_habitat.sh all

# (3) Generate Replica NVS data
bash scripts/data/generate_replica_nvs.sh all
```

### Matterport3D


To download Matterport3D dataset, please refer to the instruction in [Matterport3D](https://niessner.github.io/Matterport/).

The download script is not included here as there is a __Term of Use agreement__ for using Matterport3D data. 

However, our method **does not** require the full Matterport3D dataset. 
Users can download the data related to the task **habitat** only.

```
# Example use of the Matterport3D download script:
python download_mp.py -o data/MP3D --task_data habitat

# Unzip data
cd data/MP3D/v1/
unzip mp3d_habitat.zip
rm mp3d_habitat.zip
cd ${ROOT}
```

## Update SplaTAM

Include new/modified SplaTAM source code and configurations to SplaTAM directory.

```
bash scripts/activegamer/update_splatam.sh

```

## Running AcriveGAMER

Here we provide the script to run the full system described in the paper. 
This script also includes the upcoming [evaluation process](#evaluation).


```
# Run Replica 
bash scripts/activegamer/run_replica.sh {SceneName/all} {NUM_TRIAL} {EXP_NAME} {ENABLE_VIS}

# Run MP3D 
bash scripts/activegamer/run_mp3d.sh {SceneName/all} {NUM_TRIAL} {EXP_NAME} {ENABLE_VIS}

# examples
bash scripts/activegamer/run_replica.sh office0 1 ActiveGAMER 1
bash scripts/activegamer/run_mp3d.sh gZ6f7yhEvPG 1 ActiveGAMER 0
bash scripts/activegamer/run_replica.sh all 5 ActiveGAMER 0
```


<h2 id="evaluation"> 🔎 Evaluation  </h2>

We evaluate the reconstruction using the following metrics with a threshold of 5cm: 

- Accuracy (cm)
- Completion (cm)
- Completion ratio (%) 

We also compute the mean absolute distance, MAD (cm), between the estimated SDF distance on all vertices from the ground truth mesh. 

### Quantitative Evaluation

```
# Evaluate Replica result
bash scripts/evaluation/eval_replica_activegamer_recon.sh {SceneName/all} {TrialIndex} {EXP_NAME}
bash scripts/evaluation/eval_mp3d_activegamer_recon.sh {SceneName/all} {TrialIndex} {EXP_NAME}

# Examples
bash scripts/evaluation/eval_replica_activegamer_recon.sh office0 0 2000
bash scripts/evaluation/eval_mp3d_activegamer_recon.sh gZ6f7yhEvPG 0 5000
```


<h2 id="license"> 📜 License  </h2>

ActiveGAMER is licensed under [MIT licence](LICENSE). For the third parties, please refer to their license. 

- [HabitatSim](https://github.com/facebookresearch/habitat-sim/blob/main/LICENSE): MIT License
- [SplaTAM](https://github.com/spla-tam/SplaTAM): BSD-3-Clause License

### Modified Individual Files

The following files are adapted from third-party sources and modified. Their original licenses still apply:

- `utils/c2e_utils.py`  
Based on [py360convert/utils.py](https://github.com/sunset1995/py360convert) — MIT License
- `src/evaluation/eval_recon.py`  

<h2 id="acknowledgement"> 🤝 Acknowledgement  </h2>

We also gratefully acknowledge projects whose ideas inspired this work but whose source code is **not** included here, e.g. [gradslam](https://github.com/gradslam/gradslam), etc.



<h2 id="citation"> 📖 Citation  </h2>

```
@article{chen2025activegamer,
  title={ActiveGAMER: Active GAussian Mapping through Efficient Rendering},
  author={Chen, Liyan and Zhan, Huangying and Chen, Kevin and Xu, Xiangyu and Yan, Qingan and Cai, Changjiang and Xu, Yi},
  journal={arXiv preprint arXiv:2501.06897},
  year={2025}
}
```
