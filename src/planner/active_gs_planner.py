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

from collections import defaultdict
import math
import mmengine
import numpy as np
import os
import torch
import torch.nn.functional as F
from typing import List, Tuple

from src.planner.planner import compute_camera_pose
from src.planner.naruto_planner import NarutoPlanner
from src.utils.general_utils import InfoPrinter
from src.planner.rotation_planning import rotation_planning
from src.data.pose_loader import PoseLoader
from src.planner.rotation_planner_v2 import smoothen_trajectory_v2 as smoothen_trajectory

from third_party.splatam.utils.slam_external import calc_psnr


def remove_consecutive_duplicates(array_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Removes consecutive duplicate elements from a list of NumPy arrays.

    Args:
        array_list (list[np.ndarray]): A list of NumPy arrays, each with shape (3,).

    Returns:
        list[np.ndarray]: A new list with consecutive duplicate elements removed.
    """
    if not array_list:
        return []

    # Initialize the result list with the first element
    result = [array_list[0]]

    # Iterate through the list starting from the second element
    for i in range(1, len(array_list)):
        if not np.array_equal(array_list[i], array_list[i - 1]):
            result.append(array_list[i])

    return result


class ActiveGSPlanner(NarutoPlanner):
    def __init__(self, 
                 main_cfg    : mmengine.Config,
                 info_printer: InfoPrinter,
                 ) -> None:
        """
        Args:
            main_cfg (mmengine.Config): Configuration
            info_printer (InfoPrinter): information printer

        Attributes:
            state (str): planner state
            is_observation_done (bool): is goal pose reached
            pose_loader (PoseLoader)
            obs_poses (List[torch.Tensor])
        """
        super(ActiveGSPlanner, self).__init__(main_cfg, info_printer)
        self.device = main_cfg.general.device
        
        ### initialize planner state ###
        self.state = "stay"

        ### initialize pose loader ###
        self.pose_loader = PoseLoader(main_cfg)

        self.obs_poses = []


        ### initialize refinement observation poses ###
        refine_steps = int(2 * math.pi * 0.3 / 0.1)
        self.refine_pose_set = self.generate_circular_trajectory(0.3, 0.1, refine_steps).numpy() # refine_pose-to-center 

        ### initialize explore and refine pool ###
        self.explore_pool = defaultdict(lambda: {})
        self.refine_pool = {}
        self.planning_state = "exploration"
        self.exploration_stage = 0
        self.num_exploration_stage = self.planner_cfg.num_exploration_stage
        self.first_done_exploration = False

        ### initialize view direction sampling ###
        self.num_dir_samples = self.planner_cfg.num_dir_samples
        self.view_rot_samples = []
        self.view_rot_idx = []
        for i in range(self.num_exploration_stage):
            self.view_rot_samples.append(self.generate_rotation_samples(
                torch.from_numpy(self.planner_cfg.up_dir).to(self.device).float(),
                self.num_dir_samples[i],
            )) # 1, K, 4, 4
            self.view_rot_idx.append(torch.range(0, self.num_dir_samples[i]-1).unsqueeze(0).unsqueeze(2).to(self.device))

    def generate_circular_trajectory(self, radius: float, delta: float, steps: int) -> torch.Tensor:
        """
        Generate a circular trajectory around the current camera pose.
        Each step moves along the circle anticlockwise with a step size of delta.

        Args:
            radius (float): Radius of the circular trajectory on the XY plane.
            delta (float): Maximum step size for each movement.
            steps (int): Total number of steps for the circular trajectory.

        Returns:
            torch.Tensor: A tensor containing Nx4x4 matrices representing poses along the trajectory.
        """
        
        # Extract the initial rotation and translation from the pose
        current_pose = torch.eye(4)
        rotation = current_pose[:3, :3].clone()  # Copy of the initial rotation matrix
        initial_position = current_pose[:3, 3].clone()  # Initial position (x, y, z)
        
        # Calculate the initial point on the circle's surface (+X direction on XY plane)
        circle_start_position = initial_position.clone()
        circle_start_position[0] += radius  # Move along +X axis on XY plane


        # Initialize the list of poses with the starting pose on the circle
        init_path = self.interpolate_path(initial_position, circle_start_position, 0.1)
        trajectory = []
        for i in range(len(init_path)):
            next_pose = torch.eye(4)
            next_pose[:3, :3] = rotation
            next_pose[:3, 3] = init_path[i]
            trajectory.append(next_pose)

        # Calculate the angle step based on `delta` and `radius`
        angle_step = delta / radius  # radians per step
        angle = 0.0  # Start angle at 0 (along +X)

        for _ in range(steps + 1):
            # Update angle for the next step
            angle += angle_step  # Move anticlockwise (positive angle in radians)

            # Compute the next position on the circle using polar coordinates (X = cos, Y = sin)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            # Calculate the new position by adding to the Z coordinate of the initial position
            next_position = initial_position.clone()
            next_position[0] += x
            next_position[1] += y

            # Create the 4x4 transformation matrix for this position
            next_pose = torch.eye(4)
            next_pose[:3, :3] = rotation  # Keep the initial orientation constant
            next_pose[:3, 3] = next_position  # Update position

            # Append to the trajectory list
            trajectory.append(next_pose)

        # Initialize the list of poses with the starting pose on the circle
        last_path = self.interpolate_path(next_position, initial_position, 0.1)
        for i in range(len(last_path)):
            next_pose = torch.eye(4)
            next_pose[:3, :3] = rotation
            next_pose[:3, 3] = last_path[i]
            trajectory.append(next_pose)

        # Convert the list of poses to a tensor
        trajectory_tensor = torch.stack(trajectory)
        return trajectory_tensor

    def pose_conversion_sim2slam(self, sim_pose: torch.Tensor) -> torch.Tensor:
        """ Convert pose from Simulator system to SplaTAM system
        
        Args:
            sim_pose: Simulator pose
        
        Returns:
            slam_pose: SplaTAM pose
        """
        slam_pose = sim_pose.clone()
        slam_pose[:3, 1] *= -1
        slam_pose[:3, 2] *= -1
        slam_pose = self.sim2slam @ slam_pose
        return slam_pose

    def pose_conversion_slam2sim(self, slam_pose: torch.Tensor) -> torch.Tensor:
        """ Convert pose from SplaTAM system to Simulator system
        
        Args:
            slam_pose: SplaTAM pose
        
        Returns:
            sim_pose: Simulator pose
        """
        sim_pose = slam_pose.clone().to(self.sim2slam.device)
        sim_pose[:3, 1] *= -1
        sim_pose[:3, 2] *= -1
        sim_pose = torch.inverse(self.sim2slam) @ sim_pose
        return sim_pose
    
    def coord_conversion_slam2sim(self, slam_coords: torch.Tensor) -> torch.Tensor:
        """ Convert coordinate from SplaTAM system to Simulator system
        
        Args:
            slam_coords: [N,3], slam coordinates
        
        Returns:
            sim_coords: [N,3] Simulator pose
        """
        sim_coords = self.transform_points(torch.inverse(self.sim2slam), slam_coords)
        return sim_coords
    
    def coord_conversion_sim2slam(self, sim_coords: torch.Tensor) -> torch.Tensor:
        """ Convert coordinate from SplaTAM system to Simulator system
        
        Args:
            sim_coords: [N,3], slam coordinates
        
        Returns:
            slam_coords: [N,3] Simulator pose
        """
        slam_coords = self.transform_points(self.sim2slam, sim_coords)
        return slam_coords

    def init_data(self, 
                  sim2slam: torch.Tensor
                  ) -> None:
        """initialize data for ActiveGS planner
    
        Args:
            # bbox (List, [3,2]): bounding box corners coordinates
    
        Attributes:
            gs_z_levels (List, [N])                  : Goal Space Z-levels. if not provided, unitformly samples from Z range.
            sim2slam (torch.Tensor): simator-to-slam conversion
            up_dir_slam (np.ndarray, [3]): up direction in SplaTAM system
            # voxel_size (float)                       : voxel size
            # bbox (np.ndarray, [3,2])                 : bounding box corners coordinates
            # Nx/Ny/Nz (int)                           : bounding box sizes
            # gs_x/y/z_range (torch.Tensor, [X/Y/Z])   : goal space X/Y/Z levels
            # goal_space_pts (torch.Tensor, [X*Y*Z, 3]): goal space candidate locations. Unit: voxel
        """
        ### load config data ###
        self.gs_z_levels = self.planner_cfg.get("gs_z_levels", -1)
        if self.gs_z_levels == -1:
            raise NotImplementedError
        self.sim2slam = sim2slam

        up_dir_sim = torch.from_numpy(self.planner_cfg.up_dir).float().to(self.sim2slam.device).unsqueeze(1)
        self.up_dir_slam = (self.sim2slam[:3, :3] @ up_dir_sim)[:, 0].cpu().numpy()


        ### bounding box ###
        self.bbox = np.asarray(self.main_cfg.slam.bbox_bound)
        self.voxel_size = self.main_cfg.slam.bbox_voxel_size 


    def load_init_pose(self) -> torch.Tensor:
        """ load initial pose
    
        Returns:
            pose: [4, 4], camera-to-world pose
        """
        return self.pose_loader.load_init_pose().to(self.device)

    def check_observation_done(self):
        """ check if observation is completed at goal location 
        """
        is_obs_done = len(self.obs_poses) == 0
        return is_obs_done

    def rotation_planning_at_goal(self, 
                                   cur_pose : np.ndarray,
                                   goal_pose: np.ndarray
                                   ) -> np.ndarray:
        """ perform rotation planning
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
            goal_pose (np.ndarray, [4,4]): goal pose. Format: camera-to-world, RUB system
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system

        Attributes:
            rots (List): planned rotations. each element is (np.ndarray, [3,3])
        """
        rot = goal_pose[:3, :3]
        self.rots = rotation_planning(cur_pose[:3, :3], [rot], self.planner_cfg.max_rot_deg)

        new_pose = cur_pose.copy()
        return new_pose

    def rotating_at_goal(self, cur_pose: np.ndarray) -> np.ndarray:
        """ observing at the goal location using the observation poses in self.obs_poses. 
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        return self.rotating_at_current_loc(cur_pose)

    def observation_planning_at_goal(self, cur_pose : np.ndarray) -> np.ndarray:
        """ perform observation planning
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
            goal_pose (np.ndarray, [4,4]): goal pose. Format: camera-to-world, RUB system
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system

        Attributes:
            obs_poses (List): planned observations. each element is (np.ndarray, [3,3])
        """
        self.obs_poses = [cur_pose]
        new_pose = cur_pose.copy()
        return new_pose

    def observing_at_goal(self) -> np.ndarray:
        """ observing at the current location using the rotations in self.rots. 
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        new_pose = self.obs_poses.pop(0)
        return new_pose

    def update_state(self) -> None:
        """ update state machine for the planner
    
        Attributes:
            state (str): planner state
        """
        ##################################################
        ### Stay
        ##################################################
        if self.state == "stay":
            self.state = "planning"

        ##################################################
        ### planning
        ##################################################
        elif self.state == "planning":
            if self.planning_state == "post_refinement":
                self.state = "planning"
            else:
                self.state = "movingToGoal"

        ##################################################
        ### rotation planning at start point
        ##################################################
        elif self.state == "rotationPlanningAtStart":
            self.state = "rotatingAtStart"

        ##################################################
        ### rotating at start point
        ##################################################
        elif self.state == "rotatingAtStart":
            is_rotation_done = self.check_rotation_done()
            self.state = "movingToGoal" if is_rotation_done else "rotatingAtStart"

        ##################################################
        ### moving to goal
        ##################################################
        elif self.state == "movingToGoal":
            is_goal_reached = self.check_goal_reached()
            if is_goal_reached:
                # self.state = "rotationPlanningAtGoal"
                self.state = "planning"
            else:
                self.state = "movingToGoal"

        ##################################################
        ### rotation planning at goal location
        ##################################################
        elif self.state == "rotationPlanningAtGoal":
            self.state = "rotatingAtGoal"

        ##################################################
        ### rotating at goal location
        ##################################################
        elif self.state == "rotatingAtGoal":
            is_rotation_done = self.check_rotation_done()
            if is_rotation_done:
                if self.planning_state == "refinement":
                    self.state = "observationPlanningAtGoal"
                else:
                    self.state = "planning"
            else:
                self.state = "rotatingAtGoal"

        ##################################################
        ### observation planning at goal
        ##################################################
        elif self.state == "observationPlanningAtGoal":
            self.state = "observingAtGoal"

        ##################################################
        ### observation at goal
        ##################################################
        elif self.state == "observingAtGoal":
            is_observation_done = self.check_observation_done()
            self.state = "planning" if is_observation_done else "observingAtGoal"


    def moving_to_goal(self, 
                       cur_pose  : np.ndarray,
                       lookat_loc: np.ndarray,
                       next_loc  : np.ndarray,
                       up_dir    : np.ndarray = None
                       ) -> np.ndarray:
        """ moving to goal while looking at lookat_loc
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
            lookat_loc (np.ndarray, [3]): look-at location
            next_loc (np.ndarray, [3]): next location
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        rot = compute_camera_pose(next_loc, lookat_loc, up_dir=self.planner_cfg.up_dir if up_dir is None else up_dir, system="RDF")

        new_pose = cur_pose.copy()
        new_pose[:3, :3] = rot
        new_pose[:3, 3] = next_loc
        return new_pose

    def interpolate_path(self, A: torch.Tensor, B: torch.Tensor, step: float) -> List[torch.Tensor]:
        """
        Generates a list of intermediate points between two locations, A and B, with a given step size,
        excluding A and including B in the resulting list.

        Args:
            A (torch.Tensor): The starting location (1D tensor representing a point in n-dimensional space).
            B (torch.Tensor): The ending location (1D tensor representing a point in n-dimensional space).
            step (float): The step size between consecutive points.

        Returns:
            List[torch.Tensor]: A list of tensors representing intermediate points from A to B, excluding A and including B.
        
        Raises:
            ValueError: If A and B have different shapes or step size is non-positive.
        """
        if A.shape != B.shape:
            raise ValueError("A and B must have the same shape.")
        if step <= 0:
            raise ValueError("Step size must be a positive value.")

        ### Calculate the total distance between A and B ###
        distance = torch.norm(B - A)

        ### Calculate the number of intermediate steps needed ###
        num_steps = int(distance / step)

        ### Generate the intermediate points, excluding A and including B ###
        points = [A + (B - A) * ((i + 1) * step / distance) for i in range(num_steps)]

        ### Append B explicitly to ensure it is included ###
        if len(points) == 0 or ((points[-1] - B)!=0).any():
            points.append(B)
        
        return points

    def transform_points(self, transformation: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Transforms a set of 3D points by a 4x4 transformation matrix.

        Args:
            transformation (Tensor): A 4x4 tensor representing the transformation matrix.
            points (Tensor): An Nx3 tensor representing the set of 3D points.

        Returns:
            Tensor: An Nx3 tensor of transformed 3D points.
        """
        # Step 1: Convert points to homogeneous coordinates (Nx4)
        N = points.shape[0]
        ones = torch.ones((N, 1), dtype=points.dtype, device=points.device)  ### Add a column of ones
        points_homogeneous = torch.cat([points, ones], dim=1)  ### Concatenate to make Nx4 matrix

        # Step 2: Apply the transformation (4x4) to each point (Nx4)
        transformed_homogeneous = points_homogeneous @ transformation.T  ### Multiply by the transformation matrix
        
        # Step 3: Convert back from homogeneous to 3D by dividing by the last (homogeneous) coordinate
        transformed_points = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3].unsqueeze(1)

        return transformed_points

    def convert_occ_grid_to_sdf(self, occ_grid):
        """
    
        Args:
            occ_grid: [D,H,W]
    
        Returns:
            
    
        Attributes:
            
        """
        sdf_vol = occ_grid.clone()
        sdf_vol[sdf_vol==1] = 0 # convert occupied to surface
        sdf_vol[sdf_vol<0] = 100 # convert free space to +ve
        # sdf_vol[sdf_vol==-1] = 100 # convert free space to +ve
        return sdf_vol

    def local_path_planning_rrt(self,
                         sdf_vol : np.ndarray,
                         cur_vxl: np.ndarray,
                         goal_vxl: np.ndarray
                         ) -> Tuple:
        """ Path planning
    
        Args:
            sdf_vol (np.ndarray, [X,Y,Z]): SDF volume
            cur_vxl (np.ndarray, [4,4]) : current vxl. 
            goal_vxl (np.ndarray, [3])   : goal location. Unit : voxel
    
        Returns:
            path (List)             : each element is a Node. [GoalNode, ..., CurrentNode]
        """
        # ### Force initial SDF to be empty space ###
        # if self.step == 0:
        #     sdf_vol = sdf_vol * 0. + 100.
        
        ## run local path planner ##
        self.local_planner.start_new_plan(
            start = cur_vxl,
            goal = goal_vxl,
            sdf_map = sdf_vol
        )
        target_reachable = self.local_planner.run(use_free_space=True)

        if not(target_reachable):
            raise NotImplementedError

        ### find path ###
        path = self.local_planner.find_path()
        path = [i.get_xyz() for i in path[::-1]]
        path = torch.concat(path, dim=0)
        return path


    def path_planning(self, map: torch.Tensor, origin: torch.Tensor, start_pose: torch.Tensor, end_pose: torch.Tensor, trans_step: float, rot_step: float, voxel_size):
        """
    
        Args:
            map: [D, H, W], occupancy grid
            start_pose: [4, 4] start pose, c2w, SplaTAM coordinate system
            end_pose: [4, 4] end pose, c2w, SplaTAM coordinate system
            trans_step: translation step size, unit: metre
            rot_step: rotation step size, unit: degree
    
        Returns:
            path: [N, 4, 4] planned path, c2w, excluding start_pose and including end_pose
    
        Attributes:
            
        """
        if self.step == 0 or torch.norm(start_pose[:3, 3] - end_pose[:3, 3]) < 1e-4:
            path = self.interpolate_path(start_pose[:3, 3], end_pose[:3, 3], self.planner_cfg.trans_step_size) 
        else:
            ### transform to Simulator system ###
            start_pose_sim = self.pose_conversion_slam2sim(start_pose)
            end_pose_sim = self.pose_conversion_slam2sim(end_pose)

            ### transform to voxel space ###
            start_loc = start_pose_sim[:3, 3] - origin
            end_loc = end_pose_sim[:3, 3] - origin

            ### local path planner ###
            sdf_vol = self.convert_occ_grid_to_sdf(map)
            path = self.local_path_planning_rrt(
                                sdf_vol.detach().cpu().numpy(), 
                                (start_loc/self.voxel_size).detach().cpu().numpy(), 
                                (end_loc/self.voxel_size).detach().cpu().numpy()
                                )
            path *= self.voxel_size
            
            ### convert back to SLAM coordinates ###
            path += origin
            path = self.coord_conversion_sim2slam(path)
            path = [i for i in path[1:]]

        ##################################################
        ### FIXME: there is possible that duplicated waypoints are presented. need to debug (room2)
        ##################################################
        ### rotation planning ###
        path = [i.detach().cpu().numpy() for i in path]
        path = remove_consecutive_duplicates(path)
        
        # gravity_dir = self.up_dir_slam.copy()
        new_path = smoothen_trajectory(
            start_pose.detach().cpu().numpy(), 
            end_pose.detach().cpu().numpy(), 
            path, 
            rot_step, 
            -self.up_dir_slam
            )
        new_path = [i for i in new_path]
        return new_path

    def compute_next_state_pose(self, 
                                cur_pose       : torch.Tensor,
                                gs_slam
                                ) -> torch.Tensor:
        """ compute next state pose
    
        Args:
            cur_pose (torch.Tensor, [4,4]): current pose. Format: camera-to-world (RUB; relative pose in SplaTAM)
            gs_slam: SplaTAM
    
        Returns:
            new_pose (torch.Tensor, [4,4]): new pose. Format: camera-to-world

        Attributes:
            traversability_mask (np.ndarray, [X,Y,Z]): valid goal space mask. get updated in self.uncertainty_aware_planning_v2()
            is_goal_reachable (bool)                 : is goal reachable
            lookat_tgts (List)                       : uncertaint target observation locations to lookat. each element is (np.ndarray, [3])
            path (List)                              : each element is a Node. [GoalNode, ..., CurrentNode]
        """
        ##################################################
        ### planning
        ##################################################
        if self.state == "planning":
            planner_state = f"{self.planning_state}_{self.exploration_stage}" if self.planning_state == "exploration" else self.planning_state
            self.timer.start(f"rendering_planning_{planner_state}", "Planner")
            planning_out = self.rendering_based_planning(cur_pose, gs_slam)
            self.timer.end(f"rendering_planning_{planner_state}")
            self.goal_pose = planning_out['new_pose'].clone() # RelPose c2w at SplaTAM
            if (self.goal_pose == cur_pose).all():
                self.path = [self.goal_pose]
            else:
                self.timer.start(f"rrt_planning_{planner_state}", "Planner")
                self.path = self.path_planning(
                    self.gs_slam.explr_map.occupancy_grid,
                    self.gs_slam.explr_map.origin,
                    cur_pose, 
                    self.goal_pose, 
                    self.planner_cfg.trans_step_size, 
                    self.planner_cfg.rot_step_size,
                    self.main_cfg.slam.bbox_voxel_size
                    )
                self.timer.end(f"rrt_planning_{planner_state}")
            self.lookat_tgts = [self.goal_pose[:3, 3].detach().cpu().numpy()]
            new_pose = cur_pose

        ##################################################
        ### rotation planning at start location
        ##################################################
        elif self.state == "rotationPlanningAtStart":
            new_pose = self.rotation_planning_at_start(cur_pose.detach().cpu().numpy(), self.lookat_tgts[0], self.up_dir_slam, "RDF")

        ##################################################
        ### rotating at start location
        ##################################################
        elif self.state == "rotatingAtStart":
            new_pose = self.rotating_at_start(cur_pose.detach().cpu().numpy())

        ##################################################
        ### moving to goal 
        ##################################################
        elif self.state == "movingToGoal":
            new_pose = self.path.pop(0)
            
        ##################################################
        ### rotation planning at goal location
        ##################################################
        elif self.state == "rotationPlanningAtGoal":
            new_pose = self.rotation_planning_at_goal(cur_pose.detach().cpu().numpy(), self.goal_pose.detach().cpu().numpy())

        ##################################################
        ### rotating at goal location
        ##################################################
        elif self.state == "rotatingAtGoal":
            new_pose = self.rotating_at_goal(cur_pose.detach().cpu().numpy())

        ##################################################
        ### observation planning at goal location
        ##################################################
        elif self.state == "observationPlanningAtGoal":
            new_pose = self.observation_planning_at_goal(cur_pose.detach().cpu().numpy())

        ##################################################
        ### observation at goal location
        ##################################################
        elif self.state == "observingAtGoal":
            new_pose = self.observing_at_goal()

        ##################################################
        ### Stay
        ##################################################
        elif self.state == "stay":
            new_pose = cur_pose.clone()

        return new_pose

    def generate_rotation_samples(self, 
                                  up: torch.Tensor,
                                  K: int = 10, 
                                  ) -> torch.Tensor:
        """
        Generates 1xKx4x4 rotation matrices with K evenly distributed viewing directions.

        Args:
            K (int): The number of viewing directions to generate.
            up (torch.Tensor): The up direction tensor (typically a unit vector).

        Returns:
            torch.Tensor: An 1xKx4x4 tensor where each 4x4 matrix represents a transformation for a viewing direction.
        
        """
        N = 1

        ### Fibonacci lattice parameters for regular sampling on a sphere ###
        golden_ratio = (1 + 5**0.5) / 2
        indices = torch.arange(0, K, dtype=torch.float32)

        ### Generate evenly spaced directions on the sphere ###
        theta = 2 * torch.pi * indices / golden_ratio
        z = 1 - (2 * indices + 1) / K
        radius = torch.sqrt(1 - z ** 2)
        directions = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta), z), dim=1).to(up.device)

        ### Initialize tensor to store 1xNx4x4 transformation matrices ###
        transformations = torch.zeros((N, K, 4, 4)).to(up.device)

        ### Compute right, true_up, and forward vectors in batches ###
        forward = -directions.unsqueeze(0).expand(-1, K, -1)  ### Shape NxKx3
        up_expanded = up.unsqueeze(0).unsqueeze(0)  ### Shape: (1, 1, 3)
        up_expanded = up_expanded.expand(forward.shape[0], forward.shape[1], -1)  ### Shape: (N, K, 3)
        right = torch.cross(up_expanded, forward, dim=2)
        right = right / right.norm(dim=2, keepdim=True)  ### Normalize right vector
        true_up = torch.cross(forward, right, dim=2)  ### Compute orthogonal up vector

        ### Stack right, true_up, and forward to form rotation part of the matrices ###
        rotation_matrices = torch.stack((right, true_up, forward), dim=3)  ### Shape NxKx3x3

        ### Create the 4x4 transformation matrices ###
        transformations[:, :, :3, :3] = rotation_matrices  ### Set rotation part
        
        ### Set the bottom row to [0, 0, 0, 1] for homogeneous coordinates ###
        transformations[:, :, 3, 3] = 1
        return transformations

    def generate_candidate_poses(self, 
                                points: torch.Tensor, 
                                transform_pose: torch.Tensor = None,
                                ) -> torch.Tensor:
        """
        Generates NxKx4x4 candidate poses for N points in space with K evenly distributed viewing directions per point.

        Args:
            points (torch.Tensor): A Nx3 tensor representing the locations of N points in space.
            transform_pose (torch.Tensor): 4x4 pose that transform all created poses

        Returns:
            torch.Tensor: An NxKx4x4 tensor where each 4x4 matrix represents a transformation for a viewing direction.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input points tensor must be of shape Kx3.")

        N = points.shape[0]  ### Number of input points
        K = self.view_rot_samples[self.exploration_stage].shape[1]

        transformations = self.view_rot_samples[self.exploration_stage].repeat(N, 1, 1, 1)
        
        ### Create the 4x4 transformation matrices ###
        transformations[:, :, :3, 3] = points.unsqueeze(1).expand(-1, K, -1)  ### Set translation part
        
        # If transform_pose is provided, apply it to all generated poses
        if transform_pose is not None:
            # Ensure the transform_pose is a 4x4 tensor for matrix multiplication
            assert transform_pose.shape == (4, 4), "transform_pose must be a 4x4 tensor."
            # Apply the transformation to each pose using matrix multiplication
            transformations[:, :, :3, 1] *= -1
            transformations[:, :, :3, 2] *= -1 # RUB @ Sim
            transformations = transform_pose @ transformations

        return transformations

    def create_skybox_poses(self, 
                            points: torch.Tensor, 
                            up_direction: torch.Tensor, 
                            transform_pose: torch.Tensor = None
                            ) -> torch.Tensor:
        """
        Creates a set of poses for each point looking in 6 directions, like a skybox.

        Args:
            points (torch.Tensor): An Nx3 tensor representing the 3D positions.
            up_direction (torch.Tensor): A 3-element tensor representing the up direction of the world.
            transform_pose (torch.Tensor): 4x4 pose that transform all created poses

        Returns:
            torch.Tensor: An Nx6x4x4 tensor where each 4x4 matrix is a transformation matrix.
        """
        # Normalize the up direction
        up = F.normalize(up_direction, dim=0)  # Ensure it is a unit vector

        # Define the 6 directions to look at (forward, back, left, right, up, down)
        look_directions = torch.tensor([
            [1, 0, 0],   # +X
            [-1, 0, 0],  # -X
            [0, 1, 0],   # +Y
            [0, -1, 0],  # -Y
            [0, 0, 1],   # +Z
            [0, 0, -1]   # -Z
        ], dtype=points.dtype, device=points.device)  # 6x3 tensor

        # Prepare an output tensor (N x 6 x 4 x 4)
        N = points.shape[0]
        poses = torch.zeros((N, look_directions.shape[0], 4, 4), dtype=points.dtype, device=points.device)

        for i, direction in enumerate(look_directions):
            # view dir: -Z (backward)
            view_direction =  -F.normalize(direction, dim=0)  # Normalize the direction

            # Compute the right direction using cross product (up x view_dir)
            right_dir = torch.cross(up.expand_as(view_direction), view_direction)
            if all(right_dir == 0):
                right_dir[0] += 1.0
            else:
                right = F.normalize(right_dir, dim=0)

            # Compute the adjusted up direction using cross product (view_direction x right)
            adjusted_up = torch.cross(view_direction, right)

            # Create a rotation matrix (3x3)
            rotation_matrix = torch.stack([right, adjusted_up, view_direction], dim=1)  # 3x3 matrix

            # Create a transformation matrix (4x4) for each point
            poses[:, i, :3, :3] = rotation_matrix
            poses[:, i, :3, 3] = points
            poses[:, i, 3, 3] = 1.0

        # If transform_pose is provided, apply it to all skybox poses
        if transform_pose is not None:
            # Ensure the transform_pose is a 4x4 tensor for matrix multiplication
            assert transform_pose.shape == (4, 4), "transform_pose must be a 4x4 tensor."
            # Apply the transformation to each pose using matrix multiplication
            poses[:, :, :3, 1] *= -1
            poses[:, :, :3, 2] *= -1 # RUB @ Sim
            poses = transform_pose @ poses

        return poses

    def add_explore_pool_cand(self, cand_poses: torch.Tensor, cand_keys: torch.Tensor):
        """ add candiate poses to explore pool
    
        Args:
            cand_poses: [N, 4, 4], candidate poses
            cand_keys: [N, 4], candidate key with elements [X, Y, Z, R_i]
        """
        for i in range(cand_keys.shape[0]):
            key = tuple(cand_keys[i].cpu().numpy())
            if key not in self.explore_pool:
                self.explore_pool[key]['pose'] = cand_poses[i]
    
    def update_explore_pool_cand(self, explore_igs: torch.Tensor, cand_keys: List[Tuple]):
        """update explore pool candidates's explore_ig
    
        Args:
            explore_igs: [N], exploration information gain
            cand_keys: candidate key with elements [X, Y, Z, R_i]
            explore_thre: percentage of missing pixel thre
        """
        for i in range(len(explore_igs)):
            self.explore_pool[cand_keys[i]]['ig'] = explore_igs[i]

    def del_explore_pool_cand(self, explore_igs: torch.Tensor, cand_keys: List[Tuple], explore_thre: float):
        """delete explore pool candidates that do not need more observations
    
        Args:
            explore_igs: [N], exploration information gain
            cand_keys: candidate key with elements [X, Y, Z, R_i]
            explore_thre: percentage of missing pixel thre
        """
        rm_idx = torch.where(explore_igs < (self.img_h * self.img_w) * explore_thre)[0]
        # rm_idx = torch.where(explore_igs == 0)[0]
        for i in rm_idx:
            del self.explore_pool[cand_keys[i]]
    
    def get_explore_pool_poses(self) -> Tuple[torch.Tensor, List[Tuple]]:
        """ get exploration pool poses and keys
    
        Returns:
            cand_poses: [N, 4, 4], candidate poses
            cand_keys: candidate key with elements [X, Y, Z, R_i]
        """
        cand_poses = []
        cand_keys = []
        for key, val in self.explore_pool.items():
            cand_poses.append(val['pose'])
            cand_keys.append(key)
        cand_poses = torch.stack(cand_poses)
        return cand_poses, cand_keys

    def add_refine_pool_cand(self, kf_data: List):
        """ add keyframe candidates to refinement pool
    
        Args:
            kf_data: keyframe data
    
        Attributes:
            refine_pool: add candidates to refine_pool
            
        """
        for kf in kf_data:
            if self.step != 0:
                sim_c2w = self.pose_conversion_slam2sim(torch.inverse(kf['est_w2c']))
                kf_vxl = self.gs_slam.explr_map.transform_xyz_to_vxl(sim_c2w[:3, 3].unsqueeze(0))
                min_dist = self.gs_slam.explr_map.compute_min_distance_from_occ(
                    self.gs_slam.explr_map.occupancy_grid, 
                    kf_vxl
                    )[0]
                if min_dist * self.gs_slam.explr_map.voxel_size > self.planner_cfg.surface_dist_thre:
                    self.refine_pool[kf['id']] = kf    
            else:
                self.refine_pool[kf['id']] = kf
    
    def get_refine_pool_data(self) -> Tuple[List[Tuple], List[Tuple], torch.Tensor]:
        """ get refine pool data
    
        Returns:
            cand_data: refinement candidate data
            cand_keys: refinement candidate key
            cand_poses: refinement candidate poses
        """
        cand_data = []
        cand_keys = []
        cand_poses = []
        for key, val in self.refine_pool.items():
            cand_data.append(val)
            cand_keys.append(key)
            cand_poses.append(torch.inverse(val['est_w2c']))
        cand_poses = torch.stack(cand_poses)
        return cand_data, cand_keys, cand_poses

    def del_refine_pool_cand(self, 
                             color_igs: torch.Tensor, 
                             depth_igs: torch.Tensor, 
                             cand_keys: List[Tuple],
                             target_psnr: float = 34,
                             target_rel_depth_err: float = 0.01
                             ):
        """delete refine pool candidates that are already good enough
    
        Args:
            color_igs: [N], color information gain (PSNR)
            depth_igs: [N], depth informatoion gain (rel. depth error)
            cand_keys: candidate key with keyframe index
            target_psnr: target PSNR
            target_rel_depth_err: target relative depth error

        """
        rm_idx = torch.where((color_igs > target_psnr) * (depth_igs < target_rel_depth_err))[0]
        for i in rm_idx:
            del self.refine_pool[cand_keys[i]]

    def rendering_based_planning(self, 
                                 cur_pose,
                                 gs_slam
                                 ):
        """ Rendering-based planning (goal searching)
    
        Args:
            cur_pose (torch.Tensor, [4,4]): current pose. Format: camera-to-world (RUB; relative pose in SplaTAM)
            gs_slam: SplaTAM
    
        Returns:
            Dict: planning output
                # - path (List)             : each element is a Node. [GoalNode, ..., CurrentNode]
    
        Attributes:
        """
        new_pose = cur_pose

        ##################################################
        ### Get Exploration map
        ##################################################
        # if self.planning_state in ["exploration", "refinement"]:
        if self.planning_state == "exploration" and self.exploration_stage < self.num_exploration_stage:
            self.info_printer(f"Current state: {self.state} | {self.planning_state}: Getting New Exploration Map", self.step, self.__class__.__name__)

            ### get exploration map (explored free space @ Sim coordinate system) ###
            gs_z_levels = self.gs_z_levels[self.exploration_stage]
            xy_sampling_step = self.planner_cfg.xy_sampling_step[self.exploration_stage]
            new_free_voxels = gs_slam.explr_map.get_new_free_voxels(
                use_xyz_filter=True, 
                xy_sampling_step=xy_sampling_step,
                gs_z_levels=gs_z_levels
                )
            new_free_locs_sim = gs_slam.explr_map.origin + new_free_voxels * gs_slam.explr_map.voxel_size

            ### update previous_free_grid ###
            gs_z_levels = self.gs_z_levels[self.exploration_stage]
            xy_sampling_step = self.planner_cfg.xy_sampling_step[self.exploration_stage]
            gs_slam.explr_map.update_prev_free_voxels(
                use_xyz_filter=True, 
                xy_sampling_step=xy_sampling_step,
                gs_z_levels=gs_z_levels
                )

            ##################################################
            ### sample new candidates
            ##################################################
            ### sample poses (SplaTAM system) from free space ###
            if new_free_locs_sim.shape[0] != 0:
                new_cand_poses = self.generate_candidate_poses(
                                        new_free_locs_sim, 
                                        gs_slam.explr_map.sim2slam
                                        ).reshape(-1, 4, 4)

                free_vxl_idx_exp = new_free_voxels.unsqueeze(1).repeat(1, self.num_dir_samples[self.exploration_stage], 1)
                view_rot_idx_exp = self.view_rot_idx[self.exploration_stage].repeat(new_free_voxels.shape[0], 1, 1)
                new_cand_pose_key = torch.cat([free_vxl_idx_exp, view_rot_idx_exp], dim=-1).reshape(-1, 4)

                self.add_explore_pool_cand(new_cand_poses, new_cand_pose_key)

            ##################################################
            ### Exploration
            ##################################################
            is_explore_done = len(self.explore_pool) == 0
            
            ### FIXME: debug ###
            # is_explore_done = self.step > 50
            # if self.step > 50:
            #     self.gs_slam.print_and_save_result("exploration")
            
            ##################################################
            ### Evaluation when exploration is just done
            ##################################################
            if is_explore_done:
                if self.exploration_stage < self.num_exploration_stage:
                    self.info_printer(f"Current state: {self.state} | {self.planning_state}: Done Exploration Stage - {self.exploration_stage} , starting evaluation...", self.step, self.__class__.__name__)
                    eval_dir_suffix = f"exploration_stage_{self.exploration_stage}"
                    self.gs_slam.print_and_save_result(eval_dir_suffix, is_prune_gaussians=False, ignore_first_frame=True)
                    ### save step  ###
                    eval_dir = self.gs_slam.eval_dir + "_" + eval_dir_suffix
                    os.makedirs(eval_dir, exist_ok=True)
                    with open(os.path.join(eval_dir, "exploration_info.txt"), 'w') as f:
                        line = f"exploration_stage_{self.exploration_stage}_step: {self.step}\n"
                        f.writelines(line)

                        line = "global_keyframe: "
                        f.writelines(line)
                        
                        if self.main_cfg.slam.use_global_keyframe:
                            line = f"{sorted(self.gs_slam.global_keyframe_indices)}\n"
                            f.writelines(line)

                    self.exploration_stage += 1
                    self.planning_state = "exploration"
                
                if self.exploration_stage == self.num_exploration_stage:
                    self.info_printer(f"Current state: {self.state} | {self.planning_state}: Done All Exploration.", self.step, self.__class__.__name__)
                    if self.main_cfg.slam.use_global_keyframe:
                        self.planning_state = "post_refinement"
                    else:
                        self.planning_state = "done"
                else:
                    ### reset exploration map ###
                    gs_z_levels = self.gs_z_levels[self.exploration_stage]
                    xy_sampling_step = self.planner_cfg.xy_sampling_step[self.exploration_stage]
                    gs_slam.explr_map.prev_free_voxels = torch.empty(0, 3).to(self.device)

                # if not(self.first_done_exploration):
                #     # self.gs_slam.print_and_save_result("exploration_prune", is_prune_gaussians=True)
                #     self.gs_slam.print_and_save_result(f"exploration_{self.step:04}", is_prune_gaussians=False)
                #     self.first_done_exploration = True
                

            # if not(is_explore_done):
            else:
                self.info_printer(f"Current state: {self.state} | {self.planning_state}: Evaluate Exploration Candidate I.G.", self.step, self.__class__.__name__)
                self.planning_state = "exploration"
                ### Get EXPLORE POOL poses and keys ###
                cand_poses, cand_keys = self.get_explore_pool_poses()
                explore_igs = []

                ### compute distance between current pose and candidate poses ###
                dists = torch.norm(cand_poses[:, :3, 3] - cur_pose[:3, 3], dim=1) + 1e-6 # avoid zero dist case
                dists_sm = torch.nn.functional.softmax(dists, dim=0)
                for i, cand_pose in enumerate(cand_poses):
                    ### render data from candidate pose ###
                    img, depth, valid_mask = gs_slam.render(cand_pose)

                    ##################################################
                    ### Ignore Simulation environement incomplete region
                    # FIXME: this simulation is time consuming.
                    # However, it is not related to our method but the imperfect simulation data.
                    ##################################################
                    depth_gt = self.sim.simulate(self.pose_conversion_slam2sim(cand_pose).detach().cpu().numpy(), no_print=True)['depth']
                    valid_sim_mask = depth_gt > 0.2 # 0.0 / 0.2 is the value that ignore rendering
                    valid_mask[0][~valid_sim_mask] = True

                    _, self.img_h, self.img_w = img.shape

                    ### compute EXPLORE I.G. ###
                    explore_ig = (valid_mask==0).sum()
                    explore_igs.append(explore_ig)

                ### compute weighted exploration I.G., weighted by distance ###
                explore_igs = torch.stack(explore_igs).float()
                explore_igs_sm = torch.nn.functional.softmax(torch.log(explore_igs), dim=0)
                weighted_explore_igs = (1 - dists_sm) * explore_igs_sm
                new_pose = cand_poses[torch.argmax(weighted_explore_igs)]
                # print("Best pose px num: ", explore_igs[torch.argmax(weighted_explore_igs)])

                ### update explore pool ###
                self.update_explore_pool_cand(explore_igs, cand_keys)

                ### remove explored view from EXPLORE POOL
                self.del_explore_pool_cand(explore_igs, cand_keys, self.planner_cfg.explore_thre)
                self.info_printer(f"Current state: {self.state} [Exploration Pool: {len(self.explore_pool)}]", self.step, self.__class__.__name__)
                self.info_printer(f"                            Exploration I.G.   : {explore_igs}", self.step, self.__class__.__name__)
        
        ##################################################
        ### Refinement (NOT USED)
        ##################################################
        if self.planning_state == "refinement":
            self.info_printer(f"Current state: {self.state} | {self.planning_state}: Evaluate Refinement Candidate I.G.", self.step, self.__class__.__name__)
            self.planning_state = "refinement"
            ### get new keyframe and update prev keyframe###
            new_kfs = self.gs_slam.get_new_keyframe_idxs()
            self.gs_slam.update_prev_keyframes()

            ### update REFINE_POOL (add new keyframes to REFINE_POOL) ###
            selected_kf_list = [elem for elem, mask in zip(self.gs_slam.keyframe_list, new_kfs) if mask]
            self.add_refine_pool_cand(selected_kf_list)

            ### render poses in REFINE_POOL ###
            cand_data, cand_keys, cand_poses = self.get_refine_pool_data()
            refine_igs = []
            color_igs = []
            depth_igs = []

            ### compute distance between current pose and candidate poses ###
            dists = torch.norm(cand_poses[:, :3, 3] - cur_pose[:3, 3], dim=1) + 1e-6 # avoid zero dist case
            dists_sm = torch.nn.functional.softmax(dists, dim=0)
            for i, cand_pose in enumerate(cand_poses):
                color, depth, valid_mask = gs_slam.render(cand_pose)

                ### compute REFINE I.G. ###
                valid_depth_mask = cand_data[i]['depth'] > 0
                color_ig = calc_psnr(color*valid_depth_mask, cand_data[i]['color']*valid_depth_mask).mean()
                color_igs.append(color_ig)
                depth_ig = (torch.abs(depth*valid_depth_mask - cand_data[i]['depth']*valid_depth_mask)/(cand_data[i]['depth']+1e-8)).sum() / valid_depth_mask.sum() 
                depth_igs.append(depth_ig)

            ### compute weighted Refinement I.G., weighted by distance ###
            color_igs = torch.stack(color_igs).float()
            depth_igs = torch.stack(depth_igs).float()

            color_igs_sm = 1 - torch.nn.functional.softmax(color_igs, dim=0)
            depth_igs_sm = torch.nn.functional.softmax(depth_igs, dim=0)
            # refine_igs = color_igs_sm * depth_igs_sm
            refine_igs = color_igs_sm
            # weighted_refine_igs = (1 - dists_sm) * refine_igs
            # new_pose = cand_poses[torch.argmax(weighted_refine_igs)]
            best_key = torch.argmax(refine_igs)
            new_pose = cand_poses[best_key]

            ### remove refined views from REFINE_POOL ###
            self.del_refine_pool_cand(
                color_igs, 
                depth_igs, 
                cand_keys,
                self.planner_cfg.color_ig_thre, 
                self.planner_cfg.depth_ig_thre
                )
            self.info_printer(f"Refinement CandKey: {cand_keys}", self.step, self.__class__.__name__)
            self.info_printer(f"Refinement ColorIG: {color_igs}", self.step, self.__class__.__name__)
            self.info_printer(f"Refinement DepthIG: {depth_igs}", self.step, self.__class__.__name__)
            self.info_printer(f"Refinement best Cand: {cand_keys[best_key]} | {color_igs[best_key]} | {depth_igs[best_key]} | {refine_igs[best_key]}", self.step, self.__class__.__name__)
            self.info_printer(f"Current state: {self.state} [Refinement Pool: {len(self.refine_pool)}]", self.step, self.__class__.__name__)

            if len(self.refine_pool) == 0:
                self.planning_state = "post_refinement"

        ##################################################
        ### Post-Refinement
        ##################################################
        if self.planning_state == "post_refinement":
            if self.step % self.planner_cfg.post_refinement_eval_freq == 0:
                self.info_printer(f"Current state: {self.state} | {self.planning_state}: Evaluate Post-Refinement Candidate I.G.", self.step, self.__class__.__name__)
                self.planning_state = "post_refinement"

                ### update REFINE_POOL (add new keyframes to REFINE_POOL) ###
                if len(self.refine_pool) == 1:
                    selected_kf_list = [self.gs_slam.keyframe_list[i] for i in self.gs_slam.global_keyframe_indices]
                    self.add_refine_pool_cand(selected_kf_list)

                ### render poses in REFINE_POOL ###
                cand_data, cand_keys, cand_poses = self.get_refine_pool_data()
                refine_igs = []
                color_igs = []
                depth_igs = []

                ### compute distance between current pose and candidate poses ###
                dists = torch.norm(cand_poses[:, :3, 3] - cur_pose[:3, 3], dim=1) + 1e-6 # avoid zero dist case
                dists_sm = torch.nn.functional.softmax(dists, dim=0)
                for i, cand_pose in enumerate(cand_poses):
                    color, depth, valid_mask = gs_slam.render(cand_pose)

                    ### compute REFINE I.G. ###
                    valid_depth_mask = cand_data[i]['depth'] > 0
                    color_ig = calc_psnr(color*valid_depth_mask, cand_data[i]['color']*valid_depth_mask).mean()
                    color_igs.append(color_ig)
                    depth_ig = (torch.abs(depth*valid_depth_mask - cand_data[i]['depth']*valid_depth_mask)/(cand_data[i]['depth']+1e-8)).sum() / valid_depth_mask.sum() 
                    depth_igs.append(depth_ig)

                ### compute weighted Refinement I.G., weighted by distance ###
                color_igs = torch.stack(color_igs).float()
                depth_igs = torch.stack(depth_igs).float()

                color_igs_sm = 1 - torch.nn.functional.softmax(color_igs, dim=0)
                depth_igs_sm = torch.nn.functional.softmax(depth_igs, dim=0) # Not Used
                # refine_igs = color_igs_sm * depth_igs_sm
                refine_igs = color_igs_sm

                ### remove refined views from REFINE_POOL ###
                self.info_printer(f"Current state: {self.state} [Refinement Pool: {len(self.refine_pool)}]", self.step, self.__class__.__name__)
                self.info_printer(f"Refinement ColorIG: {color_igs}", self.step, self.__class__.__name__)
                self.info_printer(f"Refinement ColorIG [Min, Avg]: [{torch.min(color_igs).item():.2f}, {torch.mean(color_igs).item():.2f}]", self.step, self.__class__.__name__)

                # if len(self.refine_pool) == 0:
                ### When 90% of global keyframes are good, then it is done ###
                if (color_igs > self.main_cfg.slam.global_keyframe.color_thre).sum()/len(color_igs) > 0.9:
                    self.planning_state = "done"

        if self.planning_state == "done":
            self.planning_state = "done"
            self.info_printer(f"Current state: Exploration + Refinement All Done!", self.step, self.__class__.__name__)
    
        out = dict(
            new_pose = new_pose,
        )
        return out

    def main(self, 
             cur_pose       : torch.Tensor,
             gs_slam,
             ) -> torch.Tensor:
        """ ActiveGS Planner main function
    
        Args:
            cur_pose (torch.Tensor, [4,4]): current pose. Format: camera-to-world, RUB system
            gs_slam: 3DGS SLAM
    
        Returns:
            new_pose (torch.Tensor, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        self.gs_slam = gs_slam
        self.update_state()
        self.info_printer(f"Current state: {self.state}", self.step, self.__class__.__name__)
        new_pose = self.compute_next_state_pose(cur_pose, gs_slam)
        if type(new_pose) == np.ndarray:
            new_pose = torch.from_numpy(new_pose).float().to(cur_pose.device)
        return new_pose
