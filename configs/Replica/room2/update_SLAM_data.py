import numpy as np
import os

_base_ = "../../default.py"

##################################################
### General
##################################################
general = dict(
    dataset = "Replica",
    scene = "room2",
    num_iter = 2000,
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
        room_cfg        = f"{dirs['cfg_dir']}/../replica_splatam_s.py",   # SplaTAM room configuration
        # room_cfg        = f"{dirs['cfg_dir']}/../replica_splatam.py",   # SplaTAM room configuration
        enable_active_planning = False,                             # enable/disable active planning

        ### bounding box ###
        # bbox_bound = [[-2.2,2.6],[-3.4,2.1],[-1.4,2.0]],
        bbox_bound = [[-2.1,2.5],[-3.2,2],[-1.3,2.0]],
        bbox_voxel_size = 0.05,

        surface_dist_thre = 0.5,

        ### Refinement step ###
        refine_map_iter = 5,

        ### override ###
        override = dict(
            map_every = 5,
            report_global_progress_every = 20,
            tracking = dict(
                use_gt_poses=True, # Use GT Poses for Tracking
            )
        )
    )

##################################################
### Planner
##################################################
planner = dict(
    # method= "active_lang",                           # planner method [predefined_traj, active_lang]
    method = "predefined_traj",

    ### active_lang params ###
    gs_z_levels = [20, 30, 40, 50], #[20,30,40],
    num_dir_samples = 10, # viewing direction sample number
    xy_sample_step = 0.5, # Unit: meter

    surface_dist_thre = slam['surface_dist_thre'],

    ### Stop Criteria ###
    explore_thre = 0.01,
    color_ig_thre = 34,
    depth_ig_thre = 0.01,


    up_dir = np.array([0, 0, 1]), # up direction for planning pose
    use_traj_pose = True,                          # use pre-defined trajectory pose
    SLAMData_dir = os.path.join(                    # SLAM Data directory (for passive mapping or pre-defined trajectory pose)
        dirs["data_dir"], 
        "Replica", general['scene']
        ),
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

