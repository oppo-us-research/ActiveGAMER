import numpy as np
import os

_base_ = "../../default.py"

##################################################
### General
##################################################
general = dict(
    dataset = "MP3D",
    scene = "pLe4wQe7qrG",
    num_iter = 5000,
    device = 'cuda'
)

##################################################
### Directories
##################################################
dirs = dict(
    data_dir = "data/",
    result_dir = "results/",
    cfg_dir = os.path.join("configs", general['dataset'], general['scene'])
)


##################################################
### Simulator
##################################################
sim = dict(
    method = "habitat_v2"                                  # simulator method
)

if sim["method"] == "habitat_v2":
    sim.update(
        habitat_cfg = os.path.join(dirs['cfg_dir'], "habitat.py")
    )

##################################################
### SLAM
##################################################
slam = dict(
    method="splatam"                                     # SLAM backbone method
)

if slam["method"] == "splatam":
    slam.update(
        room_cfg        = f"{dirs['cfg_dir']}/../ActiveGAMER/mp3d_splatam_s.py",   # SplaTAM room configuration
        enable_active_planning = True,                             # enable/disable active planning
        dataset_eval_basedir = "data/mp3d_sim_nvs",

        ### bounding box ###
        bbox_bound = [[-2.4,9.2],[-3.8, 3.9],[-0.2,10.5]],
        bbox_voxel_size = 0.05,

        surface_dist_thre = 0.3,
        find_free_indices_bs = 1000,

        ### Refinement step ###
        # explore_map_iter = 1,
        refine_map_iter = 60,
        use_global_keyframe = True,
        global_keyframe = dict(
            completeness_thre = 0.1,
            color_thre = 34, # smaller than this thre, add to global keyframe
            depth_thre = 0.01, # larger than this thre, add to global keyframe [NOT USED]
        ),

        ### override ###
        override = dict(
            map_every = 5,
            report_global_progress_every = 5,
            tracking = dict(
                use_gt_poses=True, # Use GT Poses for Tracking
            )
        ),

        start_c2w = np.array([
            [ 1,  0,  0,  5],
            [ 0,  0,  -1,  0],
            [ 0,  1,  0,  1.75],
            [ 0,  0,  0,  1]]
            ).astype(np.float32)
    )

##################################################
### Planner
##################################################
planner = dict(
    method= "active_gs",                           # planner method [predefined_traj, active_gs]
    # method = "predefined_traj",

    ### active_gs params ###
    num_exploration_stage = 2,
    gs_z_levels = [
        [35], 
        [20, 60, 100],
        # [20, 30, 40, 50]
    ],
    num_dir_samples = [ # viewing direction sample number
        5, 
        15,
    ],

    xy_sampling_step = [
        1.0,
        0.5,
    ], # Unit: meter

    trans_step_size = 0.1, # meter
    rot_step_size = 10, # degree

    surface_dist_thre = slam['surface_dist_thre'],

    ### Stop Criteria ###
    explore_thre = 0.005,
    color_ig_thre = 34,
    depth_ig_thre = 0.01,

    post_refinement_eval_freq = 100,



    up_dir = np.array([0, 0, 1]), # up direction for planning pose
    use_traj_pose = False,                          # use pre-defined trajectory pose
    SLAMData_dir = os.path.join(                    # SLAM Data directory (for passive mapping or pre-defined trajectory pose)
        dirs["data_dir"], 
        "Replica", general['scene']
        ),

    ### RRT ###
    local_planner_method = "RRTNaruto",             # RRT method
)

if planner["local_planner_method"] == "RRTNaruto":
    planner.update(
        rrt_step_size = planner['trans_step_size'] / slam['bbox_voxel_size'], # Unit: voxel
        rrt_step_amplifier = 10,                    # rrt step amplifier to fast expansion
        rrt_maxz = 100,                             # Maximum Z-level to limit the RRT nodes. Unit: voxel
        rrt_max_iter = None,                        # maximum iterations for RRT
        rrt_z_levels = None,                        # Z levels for sampling RRT nodes. Unit: voxel. Min and Max level
        enable_eval = False,                        # enable RRT evaluation
        enable_direct_line = True,                  # enable direct connection attempt
    )

##################################################
### Visualization
##################################################
visualizer = dict(
    method = "active_gs",
    vis_rgbd        = True,                             # visualize RGB-D
    vis_rgbd_max_depth = 10

    ### mesh related ###
    # mesh_vis_freq = 500,                                # mesh save frequency
)

