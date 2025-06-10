"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import cv2
import os
import sys
sys.path.append(os.getcwd())
from tensorboardX import SummaryWriter
import torch
import numpy as np

from src.main.cfg_loader import argument_parsing, load_cfg
from src.planner import init_planner
from src.simulator import init_simulator
from src.utils.timer import Timer
from src.utils.general_utils import fix_random_seed, InfoPrinter, update_module_step
from src.visualization import init_visualizer

def write_poses_to_file(poses, filename) -> None:
    """
    Writes a list of 4x4 pose matrices to a text file, each pose written on a new line.

    Args:
        poses (list[np.ndarray]): A list of 4x4 numpy arrays representing the poses.
        filename (str): The path to the text file where the poses will be written.

    Returns:
        None
    """
    with open(filename, 'w') as file:
        for pose in poses:
            # Flatten the 4x4 matrix and format it as a single line
            pose_line = ' '.join(map(str, pose.flatten()))
            file.write(pose_line + '\n')
    print(f"Poses have been written to {filename}.")


def generate_round_trajectory(center, radius, up_axis=np.array([0, 0, 1]), num_points=36):
    """
    Generate a round trajectory around an axis in 3D space.

    Parameters:
    - center (np.array): 3D coordinates of the center of the trajectory.
    - radius (float): Radius of the trajectory.
    - up_axis (np.array): Up-axis vector, default is [0, 0, 1].
    - num_points (int): Number of points in the trajectory.

    Returns:
    - poses (list): List of tuples (position, orientation) for each point.
    """
    up_axis = up_axis / np.linalg.norm(up_axis)
    poses = []

    # Calculate the right-axis perpendicular to up_axis
    right_axis = np.cross(up_axis, np.array([0, 0, -1]))
    if np.linalg.norm(right_axis) == 0:
        right_axis = np.array([1, 0, 0])
    right_axis = right_axis / np.linalg.norm(right_axis)

    # Generate trajectory points
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        position = center + radius * (np.cos(angle) * right_axis + np.sin(angle) * np.cross(up_axis, right_axis))

        forward = center - position
        forward = forward / np.linalg.norm(forward)

        # Recalculate up to ensure alignment with up_axis
        up = up_axis - np.dot(up_axis, forward) * forward
        up = up / np.linalg.norm(up)

        right = np.cross(forward, up)

        rotation_matrix = np.vstack((right, -up, forward)).T
        # orientation = R.from_matrix(rotation_matrix).as_quat()

        pose = np.eye(4)
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = position
        poses.append(pose.astype(np.float32))

        # poses.append((position, orientation))

    return poses


def load_poses(pose_path):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w = torch.from_numpy(c2w).float()
        poses.append(c2w)
    return poses

if __name__ == "__main__":
    info_printer = InfoPrinter("ActiveGAMER")
    timer = Timer()

    ##################################################
    ### argument parsing and load configuration
    ##################################################
    info_printer("Parsing arguments...", 0, "Initialization")
    args = argument_parsing()
    info_printer("Loading configuration...", 0, "Initialization")
    main_cfg = load_cfg(args)
    main_cfg.dump(os.path.join(main_cfg.dirs.result_dir, 'main_cfg.json'))
    info_printer.update_total_step(main_cfg.general.num_iter)
    info_printer.update_scene(main_cfg.general.dataset + " - " + main_cfg.general.scene)

    ##################################################
    ### Fix random seed
    ##################################################
    info_printer("Fix random seed...", 0, "Initialization")
    fix_random_seed(main_cfg.general.seed)

    ##################################################
    ### initialize logger
    ##################################################
    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    logger = SummaryWriter(f'{log_savedir}')
    
    ##################################################
    ### initialize simulator
    ##################################################
    sim = init_simulator(main_cfg, info_printer)

    ##################################################
    ### initialize planning module
    ##################################################
    planner = init_planner(main_cfg, info_printer)
    ##################################################
    ### Run ActiveLang
    ##################################################
    ## load initial pose and convert from RUB to RDF (splatam)) ##
    c2w_slam = planner.load_init_pose() # RUB
    c2w_slam[:3, 1] *= -1
    c2w_slam[:3, 2] *= -1 # RDF
    c2w_slam_init = c2w_slam.clone() # RDF

    ## initialize exploration map in slam ##
    T_sim2slam = torch.inverse(c2w_slam_init) # RDF # transformation that takes sim-world points to slam-world-origin (i.e. first camera)
    planner.init_data(T_sim2slam)

    ##################################################
    ### initialize visualizer
    ##################################################
    visualizer = init_visualizer(main_cfg, info_printer)

    ##################################################
    ### Generate Circular motions
    ##################################################
    nvs_poses_ref = load_poses(f"data/Replica/{main_cfg.general.scene}/traj.txt")
    nvs_poses = [nvs_poses_ref[0].detach().cpu().numpy()]
    for i in [
        main_cfg.sim.center, 
        ]:
        nvs_poses += generate_round_trajectory(
            np.array(i),
            main_cfg.sim.radius,
            np.array(main_cfg.planner.up_dir),
            36*5
        )

    new_data_dir = f"data/replica_sim_nvs/{main_cfg.general.scene}/results_habitat"
    os.makedirs(new_data_dir, exist_ok=True)
    nvs_poses_slam = []
    for i in range(len(nvs_poses)):
        update_module_step(i, [sim, planner, visualizer])

        ##################################################
        ### load pose and transform pose
        ##################################################
        c2w_sim = nvs_poses[i] # RDF
        c2w_sim[:3, 1] *= -1
        c2w_sim[:3, 2] *= -1 # RUB

        c2w_slam = planner.pose_conversion_sim2slam(torch.from_numpy(c2w_sim).float().cuda()).detach().cpu().numpy()
        c2w_slam = torch.inverse(T_sim2slam.cpu()) @ c2w_slam
        nvs_poses_slam.append(c2w_slam.detach().cpu().numpy())

        ##################################################
        ### Simulation
        ##################################################
        timer.start("Simulation", "General")
        sim_out = sim.simulate(c2w_sim)
        color = sim_out['color']
        depth = sim_out['depth']
        is_too_close = (depth < 0.2).sum() / (depth.shape[0] * depth.shape[1]) > 0.1
        assert not(is_too_close), "Too many close-camera regions"
        if main_cfg.visualizer.vis_rgbd:
            visualizer.visualize_rgbd(color, depth, main_cfg.visualizer.vis_rgbd_max_depth)
        timer.end("Simulation")
        
        ##################################################
        ### save data
        ##################################################
        ### Save Depth ###
        depth_png_scale = 6553.5
        img_path = os.path.join(new_data_dir, 'depth{:06}.png'.format(i))
        depth = np.clip((depth.detach().cpu().numpy() * depth_png_scale), 0, 65535).astype(np.uint16)
        cv2.imwrite(img_path, depth)

        ### Save Depth ###
        img_path = os.path.join(new_data_dir, 'frame{:06}.jpg'.format(i))
        color = (color.cpu().numpy()*255).astype(np.uint8)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, color)

    ### Save pose ###
    write_poses_to_file(nvs_poses_slam, os.path.join(new_data_dir, "../traj.txt"))
