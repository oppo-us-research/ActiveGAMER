#!/bin/bash
##################################################
### This script is to generate Novel View Synthesis 
###  data for evaluation on Replica dataset.
##################################################

# Input arguments
scene=${1:-office0}
EXP=${2:-generate_nvs_data} # config in configs/{DATASET}/{scene}/{EXP}.py will be loaded
ENABLE_VIS=${3:-0}
GPU_ID=${4:-0}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
PROJ_DIR=${PWD}
DATASET=Replica

##################################################
### Scenes
###     choose one or all of the scenes
##################################################
scenes=(room0 room1 room2 office0 office1 office2 office3 office4)
# Check if the input argument is 'all'
if [ "$scene" == "all" ]; then
    selected_scenes=${scenes[@]} # Copy all scenes
else
    selected_scenes=($scene) # Assign the matching scene
fi

##################################################
### Main
###     Run for selected scenes for N trials
##################################################
for scene in $selected_scenes
do
        # ### run experiment ###
        CFG=configs/${DATASET}/${scene}/${EXP}.py
        python src/data/generate_Replica_NVS_data.py \
        --cfg ${CFG} \
        --seed 0 \
        --result_dir tmp/generate_data \
        --enable_vis ${ENABLE_VIS}

done
