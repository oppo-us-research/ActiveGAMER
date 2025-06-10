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
import mmengine
import numpy as np
import os
import torch

from src.visualization.visualizer import Visualizer
from src.utils.general_utils import InfoPrinter

class ActiveGSVisualizer(Visualizer):
    def __init__(self, 
                 main_cfg    : mmengine.Config,
                 info_printer: InfoPrinter
                 ) -> None:
        """
        Args:
            main_cfg (mmengine.Config): Configuration
            info_printer (InfoPrinter): information printer
    
        Attributes:
            main_cfg (mmengine.Config): configurations
            vis_cfg (mmengine.Config) : visualizer model configurations
            info_printer (InfoPrinter): information printer
            
        """
        super(ActiveGSVisualizer, self).__init__(main_cfg, info_printer)
    
    def save_rgbd(self, 
                  color: torch.Tensor, 
                  depth: torch.Tensor
                  ) -> None:
        """save RGB-D visualization
    
        Args:
            rgb (torch.Tensor, [H,W,3]): color map. Range: 0-1
            depth (torch.Tensor, [H,W]): depth map.
    
        """
        rgbd_vis = self.visualize_rgbd(color, depth, return_vis=True)
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "rgbd", f"{self.step:04}.png")
        rgbd_vis = (rgbd_vis * 255).astype(np.uint8)
        cv2.imwrite(filepath, rgbd_vis)

    def save_pose(self, pose: np.ndarray) -> None:
        """ save pose
    
        Args:
            pose: [4,4], current pose. Format: camera-to-world, RUB system
    
        """
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "pose", f"{self.step:04}.npy")
        np.save(filepath, pose)
    