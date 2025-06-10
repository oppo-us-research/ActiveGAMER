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


import habitat_sim
import matplotlib.pyplot as plt
import mmengine
import numpy as np
import quaternion
import torch
from typing import Tuple, Union, Dict

from src.layers.erp_conversions import ERPDepth2Dist
from src.simulator.simulator import Simulator
from src.simulator.habitat_utils import make_configuration, simulate_objects
from src.utils.general_utils import InfoPrinter, apply_colormap, create_class_colormap


class HabitatSim(Simulator):
    def __init__(self, 
                 main_cfg: mmengine.Config, info_printer: InfoPrinter,
                 disable_erp: bool =False, disable_pinhole: bool = False
                 ):
        """
        Args:
            main_cfg (mmengine.Config): Configuration
            info_printer (InfoPrinter): information printer
            disable_erp (bool)        : override disable_erp
            disable_pinhole (bool)    : override disable_pinhole

        Attributes:
            sim (habitat_sim.Simulator): habitat simulator
        """
        super(HabitatSim, self).__init__(main_cfg, info_printer)

        cfg = mmengine.Config.fromfile(self.sim_cfg.habitat_cfg)

        if disable_erp:
            cfg.camera.equirectangular.enable = False
        
        if cfg.camera.equirectangular.enable:
            pano_hw = tuple(cfg.camera.equirectangular.resolution_hw)
            self.erp_depth_to_erp_dist = ERPDepth2Dist(512, pano_hw, 'cuda') 
        
        if disable_pinhole:
            cfg.camera.pinhole.enable = False
        
        sim_cfg = make_configuration(cfg)
        sim = habitat_sim.Simulator(sim_cfg)
        
        if "gravity" in cfg.simulator.physics:
            sim.set_gravity(cfg.simulator.physics.gravity)
        
        if "object" in cfg:
            if cfg.object.enable:
                simulate_objects(sim, cfg.object, cfg.agent)
        sim_cfg = cfg.simulator
        sim.step_physics(1.0)
        self.sim = sim

    def simulate(self, 
                 c2w            : np.ndarray,
                 return_erp     : bool = False,
                 no_print       : bool = False,
                 return_semantic: bool = False,
                 ) -> Union[
                     Tuple[torch.Tensor, torch.Tensor], 
                     Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                     Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                     Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                     ]:
        ''' Simulate and render the RGBD image with input c2w pose
        
        Args:
            c2w            : 4x4 matrix pose in RUB coord.
            return_erp     : return ERP data
            no_print       : do not print infomation
            return_semantic: return semantic map as well if true

        Returns:
            Tuple: simulation outputs
                - color (torch.Tensor, [H,W,3])    : pinhole color. Range        : 0-1
                - depth (torch.Tensor, [H,W])      : pinhole depth
                - seman (torch.Tensor, [H,W])      : semantic map
                - erp_color (torch.Tensor, [H,W,3]): equirectangular color. Range: 0-1
                - erp_depth (torch.Tensor, [H,W])  : equirectangular distance.
                - erp_seman (torch.Tensor, [H,W])  : equirectangular semantic map.
        '''
        if not(no_print):
            self.info_printer(f"Simulating at position [{c2w[0,3]:.3f}, {c2w[1,3]:.3f}, {c2w[2,3]:.3f}]", 
                            self.step, self.__class__.__name__)
        ### simulate agent motion ###
        next_state = habitat_sim.agent.AgentState()
        
        qut = quaternion.from_rotation_matrix(c2w[:3, :3])
        trans = c2w[:3, 3]
        next_state.position = trans
        next_state.rotation = qut
        
        self.sim.agents[0].set_state(next_state)

        ### get frames ###
        obs = self.sim.get_sensor_observations()

        ### get observations ###
        color = obs.get('pinhole_color_0.0', None)
        depth = obs.get('pinhole_depth_0.0', None)
        seman = obs.get('pinhole_semantic_0.0', None)

        ### post-processing data ###
        if color is not None:
            color = color[:, :, :3] / 255.
            color = torch.from_numpy(color.astype(np.float32))
        
        if depth is not None:
            depth = torch.from_numpy(depth.astype(np.float32))

        if seman is not None:
            ### convert to colormap ###
            # colormap = create_class_colormap(100)
            # seman = apply_colormap(seman, colormap)
            # seman = seman[:, :, :3] / 255.

            seman = torch.from_numpy(seman.astype(np.float32))
        
        if return_erp:
            erp_color = obs.get('erp_color', None)
            erp_depth = obs.get('erp_depth', None)
            erp_seman = obs.get('erp_semantic', None)

            if erp_color is not None:
                erp_color = erp_color[:, :, :3] / 255.
                erp_color = torch.from_numpy(erp_color.astype(np.float32))
            
            if erp_depth is not None:
                erp_depth = torch.from_numpy(erp_depth.astype(np.float32))
                # Set invalid depths to high values. It is more convenient in many situations than keeping them zero.
                erp_depth[erp_depth==0] = 1e8   
                erp_depth = self.erp_depth_to_erp_dist(erp_depth.unsqueeze(0).unsqueeze(0).to('cuda'))
            
            if erp_seman is not None:
                ### convert to colormap ###
                # seman = apply_colormap(seman, colormap)
                # seman = seman[:, :, :3] / 255.

                seman = torch.from_numpy(seman.astype(np.float32))
            
            if return_semantic:
                return color, depth, seman, erp_color, erp_depth, erp_seman
            
            return color, depth, erp_color, erp_depth
        
        if return_semantic:
            return color, depth, seman
        else:
            return color, depth

class HabitatSimV2(HabitatSim):
    def simulate(self, 
                c2w            : np.ndarray,
                return_erp     : bool = False,
                no_print       : bool = False,
                return_semantic: bool = False,
                ) -> Dict[str, torch.Tensor]:
        ''' Simulate and render the RGBD image with input c2w pose
        
        Args:
            c2w            : 4x4 matrix pose in RUB coord.
            return_erp     : return ERP data
            no_print       : do not print infomation
            return_semantic: return semantic map as well if true

        Returns:
            out: simulation output dictionary
                - color (torch.Tensor, [H,W,3])    : pinhole color. Range        : 0-1
                - depth (torch.Tensor, [H,W])      : pinhole depth
                - seman (torch.Tensor, [H,W])      : semantic map
                - erp_color (torch.Tensor, [H,W,3]): equirectangular color. Range: 0-1
                - erp_depth (torch.Tensor, [H,W])  : equirectangular distance.
                - erp_seman (torch.Tensor, [H,W])  : equirectangular semantic map.
        '''
        out = {}

        if not(no_print):
            self.info_printer(f"Simulating at position [{c2w[0,3]:.3f}, {c2w[1,3]:.3f}, {c2w[2,3]:.3f}]", 
                            self.step, self.__class__.__name__)
        ### simulate agent motion ###
        next_state = habitat_sim.agent.AgentState()
        
        qut = quaternion.from_rotation_matrix(c2w[:3, :3])
        trans = c2w[:3, 3]
        next_state.position = trans
        next_state.rotation = qut
        
        self.sim.agents[0].set_state(next_state)

        ### get frames ###
        obs = self.sim.get_sensor_observations()

        ### get observations ###
        color = obs.get('pinhole_color_0.0', None)
        depth = obs.get('pinhole_depth_0.0', None)
        seman = obs.get('pinhole_semantic_0.0', None)

        ### post-processing data ###
        if color is not None:
            color = color[:, :, :3] / 255.
            color = torch.from_numpy(color.astype(np.float32))
        
        if depth is not None:
            depth = torch.from_numpy(depth.astype(np.float32))

        if seman is not None:
            ### convert to colormap ###
            # colormap = create_class_colormap(100)
            # seman = apply_colormap(seman, colormap)
            # seman = seman[:, :, :3] / 255.

            seman = torch.from_numpy(seman.astype(np.float32))

        ### Gather output data ###
        out.update(dict(
                color = color,
                depth = depth,
            ))

        if return_erp:
            erp_color = obs.get('erp_color', None)
            erp_depth = obs.get('erp_depth', None)
            erp_seman = obs.get('erp_semantic', None)

            if erp_color is not None:
                erp_color = erp_color[:, :, :3] / 255.
                erp_color = torch.from_numpy(erp_color.astype(np.float32))
            
            if erp_depth is not None:
                erp_depth = torch.from_numpy(erp_depth.astype(np.float32))
                # Set invalid depths to high values. It is more convenient in many situations than keeping them zero.
                erp_depth[erp_depth==0] = 1e8   
                erp_depth = self.erp_depth_to_erp_dist(erp_depth.unsqueeze(0).unsqueeze(0).to('cuda'))
            
            if erp_seman is not None:
                ### convert to colormap ###
                # seman = apply_colormap(seman, colormap)
                # seman = seman[:, :, :3] / 255.

                seman = torch.from_numpy(seman.astype(np.float32))
            
            out.update(dict(
                    erp_color=erp_color,
                    erp_depth=erp_depth,
                ))
            if return_semantic:
                out.update(dict(
                    erp_seman=erp_seman
                ))
            
        if return_semantic:
            out.update(dict(
                seman = seman
            ))
        return out
    
    def plot_sim_img(self, 
                 c2w: np.ndarray,
                 ) -> None: 
        """
    
        Args:
            c2w: camera-to-world, RUB
        """
        if type(c2w) == torch.Tensor:
            c2w = c2w.detach().cpu().numpy()
        rgb = self.simulate(c2w)['color']
        plt.imshow(rgb)
        plt.show()
    
    def plot_sim_depth(self, 
                c2w: np.ndarray,
                ) -> None: 
        """
    
        Args:
            c2w: camera-to-world, RUB
        """
        if type(c2w) == torch.Tensor:
            c2w = c2w.detach().cpu().numpy()
        depth = self.simulate(c2w)['depth']
        plt.imshow(depth)
        plt.show()
