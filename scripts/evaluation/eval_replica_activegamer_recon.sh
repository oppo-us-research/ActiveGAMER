#!/bin/bash
##################################################
### This script is to evaluate ActiveGAMER system 
###  on the Replica dataset for 3D reconstruction.
##################################################

# Input arguments
scene=${1:-office0}
num_run=${2:-1}
EXP=${3:-ActiveGAMER} # config in configs/{DATASET}/{scene}/{EXP}.py will be loaded
ENABLE_VIS=${4:-0}
GPU_ID=${5:-0}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
PROJ_DIR=${PWD}
DATASET=Replica
RESULT_DIR=${PROJ_DIR}/results/

##################################################
### Random Seed
##################################################
seeds=(0 500 1000 1500 1999)
seeds=("${seeds[@]:0:$num_run}")

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
    for i in "${!seeds[@]}"; do
        seed=${seeds[$i]}

        ### create result folder ###
        result_dir=${RESULT_DIR}/${DATASET}/$scene/${EXP}/run_${i}
        mkdir -p ${result_dir}

        ### 3D Reconstruction evaluation ###
        DASHSCENE=${scene: 0: 0-1}_${scene: 0-1}
        GT_MESH=$PROJ_DIR/data/replica_v1/${DASHSCENE}/mesh.ply
        result_dir=${RESULT_DIR}/${DATASET}/$scene/${EXP}/run_${i}

        python src/evaluation/eval_splatam_recon.py \
        --ckpt ${result_dir}/splatam/exploration_stage_0/params.npz \
        --gt_mesh ${GT_MESH} \
        --transform_traj data/Replica/${scene}/traj.txt \
        --result_dir ${result_dir}/eval_3d/exploration_stage_0

        python src/evaluation/eval_splatam_recon.py \
        --ckpt ${result_dir}/splatam/exploration_stage_1/params.npz \
        --gt_mesh ${GT_MESH} \
        --transform_traj data/Replica/${scene}/traj.txt \
        --result_dir ${result_dir}/eval_3d/exploration_stage_1

        python src/evaluation/eval_splatam_recon.py \
        --ckpt ${result_dir}/splatam/final/params.npz \
        --gt_mesh ${GT_MESH} \
        --transform_traj data/Replica/${scene}/traj.txt \
        --result_dir ${result_dir}/eval_3d/final

    done
done
