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
from matplotlib import pyplot as plt
import mmengine
import numpy as np
import torch
from typing import Union


from src.utils.general_utils import InfoPrinter

# from third_party.coslam.utils import colormap_image


#### SimpleRecon ####
def colormap_image(
        image_1hw,
        mask_1hw=None,
        invalid_color=(0.0, 0, 0.0),
        flip=True,
        vmin=None,
        vmax=None,
        return_vminvmax=False,
        colormap="turbo",
):
    """
    Colormaps a one channel tensor using a matplotlib colormap.
    Args:
        image_1hw: the tensor to colomap.
        mask_1hw: an optional float mask where 1.0 donates valid pixels.
        colormap: the colormap to use. Default is turbo.
        invalid_color: the color to use for invalid pixels.
        flip: should we flip the colormap? True by default.
        vmin: if provided uses this as the minimum when normalizing the tensor.
        vmax: if provided uses this as the maximum when normalizing the tensor.
            When either of vmin or vmax are None, they are computed from the
            tensor.
        return_vminvmax: when true, returns vmin and vmax.
    Returns:
        image_cm_3hw: image of the colormapped tensor.
        vmin, vmax: returned when return_vminvmax is true.
    """
    valid_vals = image_1hw if mask_1hw is None else image_1hw[mask_1hw.bool()]
    if vmin is None:
        vmin = valid_vals.min()
    if vmax is None:
        vmax = valid_vals.max()

    cmap = torch.Tensor(
        plt.cm.get_cmap(colormap)(
            torch.linspace(0, 1, 256)
        )[:, :3]
    ).to(image_1hw.device)
    if flip:
        cmap = torch.flip(cmap, (0,))

    h, w = image_1hw.shape[1:]

    image_norm_1hw = (image_1hw - vmin) / (vmax - vmin)
    image_int_1hw = (torch.clamp(image_norm_1hw * 255, 0, 255)).byte().long()

    image_cm_3hw = cmap[image_int_1hw.flatten(start_dim=1)
    ].permute([0, 2, 1]).view([-1, h, w])

    if mask_1hw is not None:
        invalid_color = torch.Tensor(invalid_color).view(3, 1, 1).to(image_1hw.device)
        image_cm_3hw = image_cm_3hw * mask_1hw + invalid_color * (1 - mask_1hw)

    if return_vminvmax:
        return image_cm_3hw, vmin, vmax
    else:
        return image_cm_3hw


class Visualizer():
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
        self.main_cfg = main_cfg
        self.vis_cfg = main_cfg.visualizer
        self.info_printer = info_printer

    def update_step(self, step):
        """ update step information
    
        Args:
            step (int): step size
    
        """
        self.step = step

    def visualize_rgbd(self,
                       rgb       : torch.Tensor,
                       depth     : torch.Tensor,
                       max_depth : float = 100.,
                       vis_size  : int = 320,
                       return_vis: bool = False
                       ) -> Union[None, np.ndarray]:
        """ visualiz RGB-D 
        Args:
            rgb (torch.Tensor, [H,W,3]): color map. Range: 0-1
            depth (torch.Tensor, [H,W]): depth map.
            max_depth (float)          : maximum depth value
            vis_size (int)             : image size used for visualization
            return_vis (bool)          : return visualization (OpenCV format) if True

        Returns:
            Union: 
                - image (np.ndarray, [H,W,3]): RGB-D visualization if return_vis
        """
        ## process RGB ##
        rgb = cv2.cvtColor(rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)
        rgb = cv2.resize(rgb, (vis_size, vis_size))

        ### process Depth map ###
        depth = depth.unsqueeze(0)
        mask = (depth < max_depth) * 1.0
        depth_colormap = colormap_image(depth, mask)
        depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
        depth_colormap = cv2.resize(depth_colormap, (vis_size, vis_size))

        ### display RGB-D ###
        image = np.hstack((rgb, depth_colormap))

        ### return visualization ###
        if return_vis:
            return image
        else:
            cv2.namedWindow('RGB-D', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB-D', image)
            key = cv2.waitKey(1)
