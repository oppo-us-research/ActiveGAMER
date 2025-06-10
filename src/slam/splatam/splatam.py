"""
We have reused part of SplaTAM's code in this file.
For SplaTAM License, refer to https://github.com/spla-tam/SplaTAM/blob/main/LICENSE.
"""
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import torch
import torch.nn.functional as F
import time
from importlib.machinery import SourceFileLoader
import mmengine
from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple
from tqdm import tqdm
import wandb


from src.slam.slam_model import SlamModel
from src.slam.splatam.eval_helper import eval, report_progress
from src.utils.general_utils import InfoPrinter
from src.slam.splatam.exploration_map import ExplorationMap

from third_party.splatam.utils.slam_external import calc_psnr


### original Splatam modules ###
sys.path.append("third_party/splatam")
from scripts.splatam import initialize_first_timestep, get_dataset, initialize_camera_pose, initialize_optimizer, get_loss, add_new_gaussians
from datasets.gradslam_datasets import (load_dataset_config,)
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    matrix_to_quaternion, transform_to_frame, transformed_params2rendervar, transformed_params2depthplussilhouette
)
from utils.keyframe_selection import keyframe_selection_overlap
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify
from utils.eval_helpers import report_loss#, report_progress
from utils.common_utils import save_params_ckpt, save_params


PRINT_INFO = True

class SplatamOurs(SlamModel):
    def __init__(self,
                 main_cfg: mmengine.Config, info_printer: InfoPrinter, logger: SummaryWriter
                 ) -> None:
        SlamModel.__init__(self, main_cfg, info_printer, logger)

        ### Splatam config loading and save ###
        config = SourceFileLoader(
            os.path.basename(self.slam_cfg.room_cfg), self.slam_cfg.room_cfg
        ).load_module().config
        config = self.override_config(config)
        self.config = config

        ### Init WandB ###
        if self.config['use_wandb']:
            self.wandb_time_step = 0
            self.wandb_tracking_step = 0
            self.wandb_mapping_step = 0
            self.wandb_run = wandb.init(project=config['wandb']['project'],
                                entity=config['wandb']['entity'],
                                group=config['wandb']['group'],
                                name=config['wandb']['name'],
                                config=config)

        ### Get Device ###
        self.device = torch.device(self.config["primary_device"])

        ### Load dataset config ###
        self.load_splatam_dataset_config()

        ### initialize camera parameters ###
        self.num_frames = self.main_cfg.general['num_iter']
        self.init_camera_parameters()

        ### Create directories ###
        self.results_dir = os.path.join(
            config["workdir"], config["run_name"]
        )
        self.eval_dir = os.path.join(self.results_dir, "eval")
        os.makedirs(self.eval_dir, exist_ok=True)

        ### Copy config ###
        if not self.config['load_checkpoint']:
            os.makedirs(self.results_dir, exist_ok=True)
            shutil.copy(self.slam_cfg.room_cfg, os.path.join(self.results_dir, "config.py"))
        

        # Initialize list to keep track of Keyframes
        self.keyframe_list = []
        self.keyframe_time_indices = []
        self.update_prev_keyframes()
        if self.slam_cfg.use_global_keyframe:
            self.global_keyframe_indices = []
            self.global_keyframe_time_indices = []

        # Init Variables to keep track of ground truth poses and runtimes
        self.gt_w2c_all_frames = []
        self.tracking_iter_time_sum = 0
        self.tracking_iter_time_count = 0
        self.mapping_iter_time_sum = 0
        self.mapping_iter_time_count = 0
        self.tracking_frame_time_sum = 0
        self.tracking_frame_time_count = 0
        self.mapping_frame_time_sum = 0
        self.mapping_frame_time_count = 0
        checkpoint_time_idx = 0

        ### load checkpoint ###
        if self.config['load_checkpoint']:
            # self.load_checkpoint()
            raise NotImplementedError("Checkpoint loading is not implemented yet.")
        
    def init_exploration_map(self, sim2slam: torch.tensor):
        """ initialize exploration map (grid)
    
        Args:
            sim2slam: transformation that transform points from simulation coordinate system to SplaTAM system
    
        Attributes:
            explr_map (ExplorationMap)
            
        """
        self.explr_map = ExplorationMap(
            self.slam_cfg.bbox_bound, 
            self.slam_cfg.bbox_voxel_size, 
            self.device, 
            sim2slam,
            use_xyz_filter=True, 
            xy_sampling_step=self.main_cfg.planner.xy_sampling_step[0], 
            gs_z_levels=self.main_cfg.planner.gs_z_levels[0]
            )

    def load_params(self, stage="final"):
        """ load checkpoint parameters
    
        Attributes:
            params: SplaTAM parameters
        """
        ### load self variables ###
        config = self.config
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        if checkpoint_time_idx == 0:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"{stage}/params.npz")
        else:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        self.params = params
    
    def load_checkpoint(self):
        """ load checkpoint
    
        Attributes:
            variables: Splatam variables
            gt_w2c_all_frames: GT world to camera pose
            keyframe_list: keyframe list
            params: Splatam parameters
        """
        ### load self variables ###
        config = self.config
        variables = self.variables
        dataset = self.dataset_sample
        gt_w2c_all_frames = self.gt_w2c_all_frames
        keyframe_list = self.keyframe_list

        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        if checkpoint_time_idx == 0:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params.npz")
        else:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
        
        self.variables = variables
        self.gt_w2c_all_frames = gt_w2c_all_frames
        self.keyframe_list = keyframe_list
        self.params = params

    def load_splatam_dataset_config(self):
        """
        Attributes:
            seperate_tracking_res (bool)
            seperate_densification_res (bool)
            device
            dataset_sample
            densify_dataset_sample
            tracking_dataset_sample
            tracking_color
            tracking_intrinsics
        """
        # Load Dataset config
        print("Loading Dataset ...")
        dataset_config = self.config["data"]

        ### gradslam data_fg ###
        if "gradslam_data_cfg" not in dataset_config:
            gradslam_data_cfg = {}
            gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
        else:
            gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
        
        if "ignore_bad" not in dataset_config:
            dataset_config["ignore_bad"] = False

        if "use_train_split" not in dataset_config:
            dataset_config["use_train_split"] = True

        if "densification_image_height" not in dataset_config:
            dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
            dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
            self.seperate_densification_res = False
        else:
            if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
                dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
                self.seperate_densification_res = True
            else:
                self.seperate_densification_res = False

        if "tracking_image_height" not in dataset_config:
            dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
            dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
            self.seperate_tracking_res = False
        else:
            if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
                dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
                self.seperate_tracking_res = True
            else:
                self.seperate_tracking_res = False
        
        ### obtain sample dataset  ###
        self.dataset_sample = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["desired_image_height"],
            desired_width=dataset_config["desired_image_width"],
            device=self.device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        self.dataset_eval = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=self.slam_cfg.dataset_eval_basedir,
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["desired_image_height"],
            desired_width=dataset_config["desired_image_width"],
            device=self.device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        if self.seperate_densification_res:
            self.densify_dataset_sample = get_dataset(
                config_dict=gradslam_data_cfg,
                basedir=dataset_config["basedir"],
                sequence=os.path.basename(dataset_config["sequence"]),
                start=dataset_config["start"],
                end=dataset_config["end"],
                stride=dataset_config["stride"],
                desired_height=dataset_config["densification_image_height"],
                desired_width=dataset_config["densification_image_width"],
                device=self.device,
                relative_pose=True,
                ignore_bad=dataset_config["ignore_bad"],
                use_train_split=dataset_config["use_train_split"],
            )
            # Init seperate dataloader for tracking if required
        if self.seperate_tracking_res:
            self.tracking_dataset_sample = get_dataset(
                config_dict=gradslam_data_cfg,
                basedir=dataset_config["basedir"],
                sequence=os.path.basename(dataset_config["sequence"]),
                start=dataset_config["start"],
                end=dataset_config["end"],
                stride=dataset_config["stride"],
                desired_height=dataset_config["tracking_image_height"],
                desired_width=dataset_config["tracking_image_width"],
                device=self.device,
                relative_pose=True,
                ignore_bad=dataset_config["ignore_bad"],
                use_train_split=dataset_config["use_train_split"],
            )
            tracking_color, _, tracking_intrinsics, _ = self.tracking_dataset_sample[0]
            self.tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
            self.tracking_intrinsics = tracking_intrinsics[:3, :3]

    def init_camera_parameters(self):
        """
    
        Attributes:
            params: Splatam parameters
            variables: Splatam variables
            intrinsics: camera intrinsics
            first_frame_w2c: first world to camera pose
            cam
            densify_intrinsics
            densify_cam
            tracking_cam
            
        """
        if self.seperate_densification_res:
            # Initialize Parameters, Canonical & Densification Camera parameters
            params, variables, intrinsics, first_frame_w2c, cam, \
                densify_intrinsics, densify_cam = initialize_first_timestep(self.dataset_sample, self.num_frames,
                                                                                 self.config['scene_radius_depth_ratio'],
                                                                                 self.config['mean_sq_dist_method'],
                                                                                 densify_dataset=self.densify_dataset_sample,
                                                                                 gaussian_distribution=self.config['gaussian_distribution'])                                                                                                                  
            self.densify_intrinsics = densify_intrinsics
            self.densify_cam = densify_cam
        else:
            # Initialize Parameters & Canoncial Camera parameters
            params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(self.dataset_sample, self.num_frames, 
                                                                                            self.config['scene_radius_depth_ratio'],
                                                                                            self.config['mean_sq_dist_method'],
                                                                                            gaussian_distribution=self.config['gaussian_distribution'])
            self.densify_intrinsics = intrinsics
            self.densify_cam = cam
        
        if self.seperate_tracking_res:
            self.tracking_cam = setup_camera(self.tracking_color.shape[2], self.tracking_color.shape[1], 
                                        self.tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())

        self.params = params
        self.variables = variables
        self.intrinsics = intrinsics
        self.first_frame_w2c = first_frame_w2c
        self.cam = cam

    def initialize_cam_params(self, num_frames):# num_frames: int
        '''
        original GS/Splatam create fixed size of cam params, but we don not know how many frames would be
        added during active mapping. hence, do cam params one-by-one
        '''
        params = {}
        # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
        cam_rots = np.tile([1, 0, 0, 0], (1, 1))
        cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
        params['cam_unnorm_rots'] = cam_rots
        params['cam_trans'] = np.zeros((1, 3, num_frames))

        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
        return params
    
    @torch.no_grad()
    def render(self, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ''' render rgb, mask, and depth based on a given pose
        Args:
            c2w: [4,4]. camera-to-world pose, in SplaTAM system
        
        Returns:
            im: (H,W,3) # render image
            depth: (H,W) # render depth
            mask: (H,W) valid rendering mask
        '''
        cam = self.cam
        first_frame_w2c = self.first_frame_w2c
        gt_w2c = torch.linalg.inv(c2w)
        cam_params = self.initialize_cam_params(1)
        sil_thres = self.config['mapping']['sil_thres']
        with torch.no_grad():
            # Get the ground truth pose relative to frame 0
            rel_w2c = gt_w2c
            rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
            rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
            rel_w2c_tran = rel_w2c[:3, 3].detach()
            # Update the camera parameters
            cam_params['cam_unnorm_rots'][..., 0] = rel_w2c_rot_quat
            cam_params['cam_trans'][..., 0] = rel_w2c_tran

        params = self.params
        cam_trans_og = self.params['cam_trans']
        cam_rot_og = self.params['cam_unnorm_rots']
        params['cam_trans'] = cam_params['cam_trans']
        params['cam_unnorm_rots'] = cam_params['cam_unnorm_rots']
        transformed_gaussians = transform_to_frame(params, 0,
                                                   gaussians_grad=False,
                                                   camera_grad=False)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, first_frame_w2c, 
                                                                     transformed_gaussians)
    
        im, _, _, = Renderer(raster_settings=cam)(**rendervar)
        depth_sil, _, _, = Renderer(raster_settings=cam)(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        valid_depth_mask = (depth_sil[0:1] > 0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        self.params['cam_trans'] = cam_trans_og
        self.params['cam_unnorm_rots'] = cam_rot_og
        return im, rastered_depth, valid_depth_mask

    def plot_render_depth(self, c2w: torch.Tensor):
        """ plot rendered depth at the given pose
    
        Args:
            c2w: [4,4], camera-to-world, RDF
        """
        _, depth, _ = self.render(c2w)
        depth = depth[0].detach().cpu().numpy()
        plt.imshow(depth)
        plt.show()
    
    def plot_render_rgb(self, c2w: torch.Tensor):
        """ plot rendered RGB at the given pose
    
        Args:
            c2w: [4,4], camera-to-world, RDF
        """
        im, _, _ = self.render(c2w)
        im = im.permute(1,2,0).detach().cpu().numpy()
        plt.imshow(im)
        plt.show()

    def online_recon_step(self, 
                          time_idx        : int,
                          color           : torch.Tensor,
                          depth           : torch.Tensor,
                          c2w             : torch.Tensor,
                          force_map_update: bool = False,
                          dont_add_kf: bool = False,
                          only_use_global_keyframe: bool = False,
                          ) -> List:
        ''' Run one step of the co-slam process.

        Args:
            time_idx        : Current frame step
            color           : color,        [H,W,3]
            depth           : depth map,    [H,W]
            c2w             : pose. Format: RUB camera-to-world, [4,4]
            force_map_update: run map update if true
            dont_add_kf     : don't add KF to the keyframe database if true
            only_use_global_keyframe: only use global keyframes in post-refinement stage
        
        Returns:
        '''
        self.update_gs_map(time_idx, color, depth, c2w, force_map_update, dont_add_kf, only_use_global_keyframe)
        if self.slam_cfg.enable_active_planning:
            self.update_explr_map(time_idx, depth, c2w, force_map_update)

    def update_global_keyframe_set_completeness(self, depth, c2w, thre, 
                                   time_idx, curr_gt_w2c, dont_add_kf, num_frames, force_map_update, config):
        """
    
        Args:
            
    
        Returns:
            
    
        Attributes:
            
        """
        if not(dont_add_kf):
            if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                        (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()) or force_map_update:
                with torch.no_grad():
                    ### get render depth and new region mask (valid number) ###
                    ### get new region mask ###
                    new_pixel_num = ((depth > 0)*(~self.render(c2w)[2])).sum()

                    ### determine if is global keyframe ###
                    _, h, w = depth.shape
                    new_pixel_ratio = new_pixel_num / (h * w)
                    is_global_kf = new_pixel_ratio  > thre

                    ### add the index to global keyframe index ###
                    if is_global_kf or time_idx == 0:
                        self.global_keyframe_indices.append(len(self.keyframe_list))
                        self.global_keyframe_time_indices.append(time_idx)

    def update_global_keyframe_set_quality(self, color, depth, c2w, color_thre, depth_thre,
                                   time_idx, curr_gt_w2c, dont_add_kf, num_frames, force_map_update, config):
        """
    
        Args:
            
    
        Returns:
            
    
        Attributes:
            
        """
        if time_idx in self.global_keyframe_time_indices:
            return 
        
        if not(dont_add_kf):
            if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                        (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()) or force_map_update:
                with torch.no_grad():
                    render_color, render_depth, _ = self.render(c2w)
                    valid_depth_mask = depth > 0
                    color_ig = calc_psnr(render_color*valid_depth_mask, color*valid_depth_mask).mean()
                    # depth_ig = (torch.abs(render_depth*valid_depth_mask - depth*valid_depth_mask)/(depth+1e-8)).sum() / valid_depth_mask.sum() 

                    ### update is_global_kf condition ###
                    is_global_kf = color_ig < color_thre

                    ### add the index to global keyframe index ###
                    if is_global_kf or time_idx == 0:
                        self.global_keyframe_indices.append(len(self.keyframe_list))
                        self.global_keyframe_time_indices.append(time_idx)

    def update_global_keyframe_set_quality_rel(self, ):
        """
    
        Args:
            
    
        Returns:
            
    
        Attributes:
            
        """
        ### Render All KFs, excluding last few (5) keyframes (due to overfitting) ###
        color_igs = []
        for kf in self.keyframe_list[:-5]:
            c2w = torch.inverse(kf['est_w2c']) # c2w
            color, depth, valid_mask = self.render(c2w)

            ### compute REFINE I.G. ###
            valid_depth_mask = kf['depth'] > 0.2 # FIXME: 0.2 is near culling range
            color_ig = calc_psnr(color*valid_depth_mask, kf['color']*valid_depth_mask).mean()
            color_igs.append(color_ig)
        
        if len(color_igs) > 0:
            ### compute weighted Refinement I.G., weighted by distance ###
            color_igs = torch.stack(color_igs).float()

            ### Rank based on PSNR ###
            color_thre = torch.quantile(color_igs, self.slam_cfg.global_keyframe.quality_perc_thre/100.)
            kf_idxs = torch.where(color_igs <= color_thre)[0]

            ### add low quality frames to Global KF (if they are not there yet) ###
            new_kf = [elem.item() for elem in kf_idxs if elem not in self.global_keyframe_indices]
            self.global_keyframe_indices.extend(new_kf)
            new_kf_time_indices = [self.keyframe_time_indices[i] for i in new_kf]
            self.global_keyframe_time_indices.extend(new_kf_time_indices)

    @torch.no_grad()
    def update_explr_map(self,
                           time_idx        : int,
                           depth           : torch.Tensor,
                           c2w             : torch.Tensor,
                           force_map_update: bool = False,
                         ) -> List         : 
        ''' Run one step of the co-slam process.

        Args:
            time_idx        : Current frame step
            depth           : depth map,    [H,W]
            c2w             : pose. Format: RUB camera-to-world, [4,4]
            force_map_update: run map update if true
        
        Attributes:
            explr_map: update exploration map
        '''
        config = self.config
        depth = depth.to(self.device)
        c2w = c2w.to(self.device)
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0 or force_map_update:
            self.explr_map.update_from_depth_map(
                depth, 
                self.intrinsics, 
                torch.inverse(c2w),
                self.slam_cfg.surface_dist_thre,
                self.slam_cfg.get("find_free_indices_bs", 10000)
                )
            
            ## FIXME: debug visualization ##
            # self.explr_map.visualize(time_idx, in_slam_world=False)
            # ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            # save_params_ckpt(self.params, ckpt_output_dir, time_idx)

    def update_gs_map(self, 
                          time_idx: int,
                          color   : torch.Tensor,
                          depth   : torch.Tensor,
                          c2w     : torch.Tensor,
                          force_map_update: bool = False,
                          dont_add_kf: bool = False,
                          only_use_global_keyframe: bool = False,
                          ) -> List:
        ''' Run one step of the splatam process. Update GS

        Args:
            time_idx        : Current frame step
            color           : color,        [H,W,3]
            depth           : depth map,    [H,W]
            c2w             : pose. Format: RUB camera-to-world, [4,4]
            force_map_update: run map update if true
            dont_add_kf     : don't add KF to the keyframe database if true
            only_use_global_keyframe: only use global keyframes in post-refinement stage
        '''
        ### get self variables ###
        params = self.params
        variables = self.variables
        intrinsics = self.intrinsics
        first_frame_w2c = self.first_frame_w2c
        cam = self.cam
        seperate_densification_res = self.seperate_densification_res
        if seperate_densification_res:
            densify_intrinsics = self.densify_intrinsics
            densify_cam = self.densify_cam
        config = self.config
        gt_w2c_all_frames = self.gt_w2c_all_frames
        if self.config['use_wandb']:
            wandb_run = self.wandb_run
            wandb_mapping_step = self.wandb_mapping_step
            wandb_time_step = self.wandb_time_step
        eval_dir = self.eval_dir
        seperate_tracking_res = self.seperate_tracking_res
        if seperate_tracking_res:
            tracking_cam = self.tracking_cam
            tracking_intrinsics = self.tracking_intrinsics
        keyframe_list = self.keyframe_list
        num_frames = self.num_frames
        keyframe_time_indices = self.keyframe_time_indices


        ### Process poses ###
        gt_w2c = torch.linalg.inv(c2w)


        # Process RGB-D Data
        color = color.permute(2, 0, 1)
        color = color.to(self.device)
        depth = depth.unsqueeze(0)
        depth = depth.to(self.device)
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx


        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        
        # # Initialize Data for Tracking
        if seperate_tracking_res:
            ### Load tracking data ###
            tracking_h, tracking_w = self.config['data']["tracking_image_height"], self.config['data']["tracking_image_width"]
            tracking_color = F.interpolate(color.unsqueeze(0), (tracking_h, tracking_w), mode='bilinear')[0]
            tracking_depth = F.interpolate(depth.unsqueeze(0), (tracking_h, tracking_w), mode='nearest')[0]
            
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])

        ##################################################
        ### Tracking
        ##################################################
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                   plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter)
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                            wandb_run=wandb_run, wandb_step=wandb_tracking_step, wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                self.tracking_iter_time_sum += iter_end_time - iter_start_time
                self.tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()
        self.tracking_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_frame_time_count += 1

        if (time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0) and not config['tracking']['use_gt_poses']:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        ##################################################
        ### update global keyframe
        ##################################################
        if self.slam_cfg.use_global_keyframe and not(only_use_global_keyframe):
            self.update_global_keyframe_set_completeness(
                depth, c2w, 
                self.slam_cfg.global_keyframe.completeness_thre, 
                time_idx, curr_gt_w2c, dont_add_kf, num_frames, force_map_update, config
            )

        ##################################################
        ### Densification & KeyFrame-based Mapping
        ##################################################
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0 or force_map_update:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:
                    # resize RGBD frames for densification
                    densify_h, densify_w = self.config['data']["densification_image_height"], self.config['data']["densification_image_width"]
                    densify_color = F.interpolate(color.unsqueeze(0), (densify_h, densify_w), mode='bilinear')[0]
                    densify_depth = F.interpolate(depth.unsqueeze(0), (densify_h, densify_w), mode='nearest')[0]

                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians(params, variables, densify_curr_data, 
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'], config['gaussian_distribution'])
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran

                ##################################################
                ### Select Keyframes for Mapping
                ##################################################
                ### overlap keyframes ###
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                if PRINT_INFO:
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")
                    if self.slam_cfg.use_global_keyframe:
                        global_keyframe_time_indices = [frame_idx for frame_idx in self.global_keyframe_time_indices if frame_idx != time_idx]
                        print(f"\nGlobal Keyframes at Frame {time_idx}: {global_keyframe_time_indices}")
                
            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False) 

            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()

                ##################################################
                ### frame selection for map update
                ##################################################
                ### Overlap Keyframe ###
                # Randomly select a frame until current time step amongst keyframes

                if only_use_global_keyframe or (self.slam_cfg.use_global_keyframe and iter > num_iters_mapping // 2):
                    # selected_keyframes = [i for i in range(len(self.global_keyframe_indices))]
                    ### Global Keyframe ###
                    # rand_idx = np.random.randint(0, len(selected_keyframes))
                    if len(self.global_keyframe_indices) == 1:
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                    else:
                        selected_rand_keyframe_idx = np.random.choice(self.global_keyframe_indices[:-1])
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                else:
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    if selected_rand_keyframe_idx == -1:
                        # Use Current Frame Data
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                    else:
                        # Use Keyframe Data
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']

                
                iter_gt_w2c = self.gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                # Loss for current frame
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True)
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                # Backprop
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                self.mapping_iter_time_sum += iter_end_time - iter_start_time
                self.mapping_iter_time_count += 1
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            self.mapping_frame_time_sum += mapping_end_time - mapping_start_time
            self.mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            eval_dir=self.eval_dir,
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')


        ##################################################
        ### update global keyframe
        ##################################################
        if self.slam_cfg.use_global_keyframe and not(only_use_global_keyframe):
            quality_method = self.slam_cfg.global_keyframe.get("quality_method", "absolute")
            if quality_method == "absolute":
                self.update_global_keyframe_set_quality(
                    color, depth, c2w, 
                    self.slam_cfg.global_keyframe.color_thre, 
                    self.slam_cfg.global_keyframe.depth_thre, 
                    time_idx, curr_gt_w2c, dont_add_kf, num_frames, force_map_update, config
                )
            elif quality_method == "relative":
                if time_idx > 0 and time_idx % self.slam_cfg.global_keyframe.quality_freq == 0:
                    self.update_global_keyframe_set_quality_rel()
            else:
                raise NotImplementedError
        
        # Add frame to keyframe list
        if not(dont_add_kf):
            if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                        (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()) or force_map_update:
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Initialize Keyframe Info
                    curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                    # Add to keyframe list
                    keyframe_list.append(curr_keyframe)
                    keyframe_time_indices.append(time_idx)

                    
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if config['use_wandb']:
            self.wandb_time_step += 1

        torch.cuda.empty_cache()
        
        ##################################################
        ### update self variables
        ##################################################
        self.params = params
        self.variables = variables
        self.intrinsics = intrinsics
        self.first_frame_w2c = first_frame_w2c
        self.cam = cam
        self.seperate_densification_res = seperate_densification_res
        if self.seperate_densification_res:
            self.densify_intrinsics = densify_intrinsics
            self.densify_cam = densify_cam
        self.config = config
        self.gt_w2c_all_frames = gt_w2c_all_frames
        if self.config['use_wandb']:
            self.wandb_run = wandb_run
            self.wandb_mapping_step = wandb_mapping_step
            self.wandb_time_step = wandb_time_step
        self.eval_dir = eval_dir
        self.seperate_tracking_res = seperate_tracking_res
        if self.seperate_tracking_res:
            self.tracking_cam = tracking_cam
            self.tracking_intrinsics = tracking_intrinsics
        
        self.keyframe_list = keyframe_list
        self.num_frames = num_frames
        self.keyframe_time_indices = keyframe_time_indices

    def print_and_save_result(self, eval_dir_suffix="", is_prune_gaussians=False, ignore_first_frame=False):
        """ evaluate rendering results and save result
        """
        ### get self variables ###
        params = self.params.copy()
        variables = self.variables.copy()
        intrinsics = self.intrinsics
        first_frame_w2c = self.first_frame_w2c
        tracking_iter_time_sum = self.tracking_iter_time_sum
        tracking_frame_time_sum = self.tracking_frame_time_sum
        tracking_iter_time_count = self.tracking_iter_time_count
        tracking_frame_time_count = self.tracking_frame_time_count
        mapping_iter_time_sum = self.mapping_iter_time_sum
        mapping_frame_time_sum = self.mapping_frame_time_sum
        mapping_iter_time_count = self.mapping_iter_time_count
        mapping_frame_time_count = self.mapping_frame_time_count
        config = self.config
        if self.config['use_wandb']:
            wandb_run = self.wandb_run
        dataset_config = self.config['data']
        gt_w2c_all_frames = self.gt_w2c_all_frames
        keyframe_time_indices = self.keyframe_time_indices
        # dataset = self.dataset_sample
        dataset = self.dataset_eval
        num_frames = self.num_frames
        eval_dir = self.eval_dir + "_" + eval_dir_suffix if eval_dir_suffix else self.eval_dir

        ### prune gaussians ###
        if is_prune_gaussians:
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False) 
            params, variables = prune_gaussians(params, variables, optimizer, 0, config['mapping']['pruning_dict'])


        # Compute Average Runtimes
        if tracking_iter_time_count == 0:
            tracking_iter_time_count = 1
            tracking_frame_time_count = 1
        if mapping_iter_time_count == 0:
            mapping_iter_time_count = 1
            mapping_frame_time_count = 1
        tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
        tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
        mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
        mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
        print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
        print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
        print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
        print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
        if config['use_wandb']:
            wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                        "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                        "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                        "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                        "Final Stats/step": 1})
        
        # Evaluate Final Parameters
        with torch.no_grad():
            if config['use_wandb']:
                eval(dataset, params, len(dataset), eval_dir, sil_thres=config['mapping']['sil_thres'],
                    wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'],
                    ignore_first_frame=ignore_first_frame)
            else:
                eval(dataset, params, len(dataset), eval_dir, sil_thres=config['mapping']['sil_thres'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'],
                    ignore_first_frame=ignore_first_frame)

        # Add Camera Parameters to Save them
        params['timestep'] = variables['timestep']
        params['intrinsics'] = intrinsics.detach().cpu().numpy()
        params['w2c'] = first_frame_w2c.detach().cpu().numpy()
        params['org_width'] = dataset_config["desired_image_width"]
        params['org_height'] = dataset_config["desired_image_height"]
        params['gt_w2c_all_frames'] = []
        for gt_w2c_tensor in gt_w2c_all_frames:
            params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
        params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
        params['keyframe_time_indices'] = np.array(keyframe_time_indices)
        
        # Save Parameters
        results_dir = os.path.join(self.results_dir, eval_dir_suffix) if eval_dir_suffix else self.results_dir
        save_params(params, results_dir)

    def eval_result(self, eval_dir_suffix="", ignore_first_frame = False, save_frames=False):
        """ evaluate rendering results
            
        """
        ### get self variables ###
        params = self.params
        config = self.config
        if self.config['use_wandb']:
            wandb_run = self.wandb_run
        # dataset = self.dataset_sample
        dataset = self.dataset_eval
        num_frames = self.num_frames
        # eval_dir = self.eval_dir
        eval_dir = self.eval_dir + "_" + eval_dir_suffix if eval_dir_suffix else self.eval_dir

        # Evaluate Final Parameters
        with torch.no_grad():
            if config['use_wandb']:
                eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                    wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'], ignore_first_frame=ignore_first_frame)
            else:
                eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'], ignore_first_frame=ignore_first_frame, save_frames=save_frames)
        return
    
    def update_dict_recursive(self, dict1, dict2):
        """
        Recursively updates the values of dict1 with the values from dict2.
        If a key in dict2 corresponds to a nested dictionary, it recursively updates dict1.

        Args:
            dict1 (Dict[Any, Any]): The dictionary to be updated.
            dict2 (Dict[Any, Any]): The dictionary with the new values.

        Returns:
            Dict[Any, Any]: The updated dict1 with values from dict2.
        """
        for key, value in dict2.items():
            if key not in dict1:
                ### Raise an error if key in dict2 is not found in dict1
                raise KeyError(f"Key '{key}' not found in dict1.")
    
            if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
                ### If both dict1[key] and dict2[key] are dictionaries, recursively update them
                dict1[key] = self.update_dict_recursive(dict1[key], value)
            else:
                ### Otherwise, update the value in dict1 with the value from dict2
                dict1[key] = value
        return dict1

    def override_config(self, config: Dict) -> Dict:
        """ override configs 
        """
        ### override from main_cfg ###
        config["data"]["sequence"] = self.main_cfg.general.scene
        config["workdir"] = os.path.join(self.main_cfg.dirs.result_dir, "splatam")
        config["run_name"] = ""
        config["data"]['gradslam_data_cfg'] = os.path.join("third_party/splatam", config["data"]['gradslam_data_cfg'])

        ## specific override ##
        config = self.update_dict_recursive(config, self.main_cfg.slam.override)

        ### Spltam config update ###
        # Print Config
        print("Loaded Config:")
        if "use_depth_loss_thres" not in config['tracking']:
            config['tracking']['use_depth_loss_thres'] = False
            config['tracking']['depth_loss_thres'] = 100000
        if "visualize_tracking_loss" not in config['tracking']:
            config['tracking']['visualize_tracking_loss'] = False
        if "gaussian_distribution" not in config:
            config['gaussian_distribution'] = "isotropic"
        print(f"{config}")

        return config
    
    def update_prev_keyframes(self):
        """ update last stored keyframe indexs
    
        Attributes:
            prev_keyframe_idxs (List): last stored keyframe indexs
            
        """
        self.prev_keyframe_idxs = self.keyframe_time_indices.copy()

    def get_new_keyframe_idxs(self) -> torch.Tensor:
        """ get new keyframe indexs
    
        Returns:
            new_kf: [N], mask for new keyframes
        """
        prev_kf_idxs = torch.tensor(self.prev_keyframe_idxs)
        new_kf_idxs = torch.tensor(self.keyframe_time_indices)

        # Flatten each row into a single unique value (row-wise hashing)
        prev_flat = prev_kf_idxs.view(-1, 1)
        new_flat = new_kf_idxs.view(1, -1)

        # Find elements in new_kf_idxs that don't match any in prev_kf_idxs
        mask = (prev_flat == new_flat).any(dim=0)
        unique_new_kf_mask = ~mask

        return unique_new_kf_mask

