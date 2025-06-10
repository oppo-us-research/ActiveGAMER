#!/bin/bash
##################################################
### This script is to generate Novel View Synthesis 
###  data for evaluation on MP3D dataset.
##################################################

# Input arguments
scene=${1:-GdvgFV5R1Z5}
EXP=${2:-generate_nvs_data} # config in configs/{DATASET}/{scene}/{EXP}.py will be loaded
ENABLE_VIS=${3:-0}
GPU_ID=${4:-0}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
PROJ_DIR=${PWD}
DATASET=MP3D

##################################################
### Scenes
###     choose one or all of the scenes
##################################################
scenes=(GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG YmJkqBEsHnH)
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
        python src/data/generate_nvs_data.py \
        --cfg ${CFG} \
        --seed 0 \
        --result_dir tmp/generate_data \
        --enable_vis ${ENABLE_VIS}

done
