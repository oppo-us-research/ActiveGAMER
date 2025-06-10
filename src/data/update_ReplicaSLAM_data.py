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
    ### initialize visualizer
    ##################################################
    visualizer = init_visualizer(main_cfg, info_printer)

    for i in range(main_cfg.general.num_iter):
        update_module_step(i, [sim, planner, visualizer])

        ##################################################
        ### load pose and transform pose
        ##################################################
        if main_cfg.planner.method == "predefined_traj":
            c2w_sim = planner.update_pose(torch.eye(4), i).numpy() # RUB

        ##################################################
        ### Simulation
        ##################################################
        timer.start("Simulation", "General")
        sim_out = sim.simulate(c2w_sim)
        color = sim_out['color']
        depth = sim_out['depth']
        if main_cfg.visualizer.vis_rgbd:
            visualizer.visualize_rgbd(color, depth, main_cfg.visualizer.vis_rgbd_max_depth)
        timer.end("Simulation")
        
        ##################################################
        ### save data
        ##################################################
        new_data_dir = f"data/Replica/{main_cfg.general.scene}/results_habitat"
        os.makedirs(new_data_dir, exist_ok=True)

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
