import torch
import numpy as np
import open3d as o3d
from typing import Tuple

class ExplorationMap:
    def __init__(self, bounding_box, voxel_size, device='cpu', transform=None, use_xyz_filter = None, xy_sampling_step = None, gs_z_levels = None):
        """
        Initialize the ExplorationMap with a bounding box, voxel size, device, and optional transform.

        Parameters:
            bounding_box (list)               : [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            voxel_size (float)                : Size of each voxel in the grid
            device (str)                      : The device to store the tensors ('cpu' or 'cuda')
            transform (torch.Tensor, optional): 4x4 transformation matrix to reposition the origin (sim2slam)
        """
        self.bounding_box = bounding_box
        self.voxel_size = voxel_size
        self.device = device
        self.sim2slam = transform.to(self.device) if transform is not None else torch.eye(4, device=device)
        self.slam2sim = torch.inverse(self.sim2slam)
        self.occupancy_grid, self.origin = self.create_occupancy_grid() # self.origin in Sim Space
        self.gs_z_levels = gs_z_levels 
        self.update_prev_free_voxels(
            use_xyz_filter=use_xyz_filter, xy_sampling_step=xy_sampling_step, gs_z_levels=gs_z_levels
            )

    def create_occupancy_grid(self):
        """
        Create a 3D occupancy grid based on the bounding box and voxel size.

        Returns:
            occupancy_grid (torch.Tensor): A 3D tensor representing the occupancy grid
            origin (torch.Tensor)        : The origin point of the occupancy grid in world coordinates
        """
        # Compute the dimensions of the occupancy grid
        min_bound = torch.tensor([self.bounding_box[0][0], self.bounding_box[1][0], self.bounding_box[2][0]], device=self.device)
        max_bound = torch.tensor([self.bounding_box[0][1], self.bounding_box[1][1], self.bounding_box[2][1]], device=self.device)
        grid_dimensions = ((max_bound - min_bound) / self.voxel_size).ceil().long()

        # Create an empty occupancy grid initialized to 0 (unoccupied)
        occupancy_grid = torch.zeros(*grid_dimensions, dtype=torch.float32, device=self.device, requires_grad=False)
        
        # Return the occupancy grid and the origin (minimum bound)
        return occupancy_grid, min_bound

    def mark_voxel_occupied(self, indices):
        """
        Mark a voxel as occupied in the occupancy grid.

        Parameters:
            indices (tuple): The indices of the voxel to mark as occupied
        """
        self.occupancy_grid[indices] = 1.0

    def get_world_coordinates_from_grid(self, 
                                        value: float = None, 
                                        in_slam_world: bool = False
                                        ) -> torch.Tensor:
        """
        Get the world coordinates of voxels in the grid.

        Parameters:
            value: If specified, return coordinates of voxels with this value

        Returns:
            world_coords: Coordinates in the world frame, shape (N, 3)
        """
        if value is None:
            indices = torch.nonzero(self.occupancy_grid, as_tuple=False)
        else:
            indices = torch.nonzero(self.occupancy_grid == value, as_tuple=False)
        
        # Convert voxel indices to world coordinates
        world_coords_sim = self.origin + indices * self.voxel_size
        N = world_coords_sim.shape[0]
        homogeneous_coords = torch.cat((world_coords_sim, torch.ones(N, 1, device=self.device)), dim=1)
        world_coords = torch.mm(homogeneous_coords, self.sim2slam.T)[:, :3] if in_slam_world else homogeneous_coords[:, :3] 
        return world_coords

    def transform_xyz_to_vxl(self, xyz: torch.Tensor):
        """
    
        Args:
            xyz: [N,3], XYZ in Simulator coordinate system
    
        Returns:
            vxl: [N,3], voxel indices in voxel space
    
        Attributes:
            
        """
        return (xyz - self.origin) / self.voxel_size

    def compute_min_distance_from_occ(self, grid: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        """
        Compute the closest distance to the occupied voxel to each point in a tensor of query points in a 3D occupancy grid,

        Args:
            grid (torch.Tensor): A 3D occupancy grid where occupied voxels are marked by 1, and free space by 0 or -1.
            query_points (torch.Tensor): A tensor of shape (M, 3), where each row represents a query point (x, y, z).

        Returns:
            min_distance (torch.Tensor): min distance to the occupied grid
        """
        ### Step 1: Get coordinates of all occupied voxels in the grid ###
        occupied_voxel_coords = torch.nonzero(grid == 1, as_tuple=False)  ### Shape (num_occupied, 3)

        ### initialization case ###
        if occupied_voxel_coords.size(0) == 0:
            return query_points, query_points.new_empty((0, 3))

        ### Step 2: Expand query points and occupied voxel coordinates for broadcasting ###
        query_points_exp = query_points.unsqueeze(1).float()  ### Shape (M, 1, 3)
        occupied_voxel_coords_exp = occupied_voxel_coords.unsqueeze(0)  ### Shape (1, num_occupied, 3)

        ### Step 3: Calculate squared Euclidean distances between each query point and each occupied voxel ###
        distances = torch.norm(query_points_exp - occupied_voxel_coords_exp, dim=2)  ### Shape (M, num_occupied)

        ### Step 4: Find the minimum distance and the corresponding nearest occupied voxel for each query point ###
        min_distances, _ = distances.min(dim=1)  ### Shape (M,)
        return min_distances

    @torch.no_grad()
    def find_free_indices(self, grid: torch.Tensor, query_points: torch.Tensor, dist_thre: float = 0.5, batch_size: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Finds free region indices by computing the closest occupied voxel to each point in a tensor of query points in a 3D occupancy grid,
        along with the distance to each closest occupied voxel.

        Args:
            grid (torch.Tensor): A 3D occupancy grid where occupied voxels are marked by 1, and free space by 0 or -1.
            query_points (torch.Tensor): A tensor of shape (M, 3), where each row represents a query point (x, y, z).
            dist_thre (float): distance from the surface threshold

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor of shape (M, 3) with the coordinates of the truncated free regions.
                - A tensor of shape (M, 3) with the coordinates of the neglected free regions.
        """
        ### Step 1: Get coordinates of all occupied voxels in the grid ###
        occupied_voxel_coords = torch.nonzero(grid == 1, as_tuple=False)  ### Shape (num_occupied, 3)

        ### initialization case ###
        if occupied_voxel_coords.size(0) == 0:
            return query_points, query_points.new_empty((0, 3))

        ### Step 2: Expand query points and occupied voxel coordinates for broadcasting ###
        query_points_exp = query_points.unsqueeze(1).float()  ### Shape (M, 1, 3)
        occupied_voxel_coords_exp = occupied_voxel_coords.unsqueeze(0)  ### Shape (1, num_occupied, 3)

        ### Step 3: Calculate squared Euclidean distances between each query point and each occupied voxel ###
        min_distances = []
        # batch_size = 10000
        num_repeat = query_points.shape[0] // batch_size
        num_repeat = num_repeat + 1 if num_repeat * batch_size < query_points.shape[0] else num_repeat
        if num_repeat > 0:
            for i in range(num_repeat):
                dist = torch.norm(query_points_exp[batch_size*i:batch_size*(i+1)] - occupied_voxel_coords_exp, dim=2)
                min_distances.append(dist.min(dim=1)[0])
            min_distances = torch.cat(min_distances, dim=0)
        else:
            distances = torch.norm(query_points_exp - occupied_voxel_coords_exp, dim=2)  ### Shape (M, num_occupied)

            ### Step 4: Find the minimum distance and the corresponding nearest occupied voxel for each query point ###
            min_distances, _ = distances.min(dim=1)  ### Shape (M,)

        ### Step 5: find the indices for truncated free region and neglected free regions ###
        valid_free_indices_mask = (min_distances * self.voxel_size) > dist_thre
        truncated_free_indices = query_points[valid_free_indices_mask]
        neglected_free_indices = query_points[~valid_free_indices_mask]
        return truncated_free_indices, neglected_free_indices

    @torch.no_grad()
    def update_from_depth_map(self, 
                              depth_map : torch.Tensor,
                              intrinsics: torch.Tensor,
                              extrinsics: torch.Tensor,
                              surface_dist_thre: float,
                              find_free_indices_bs: int = 10000
                              ) -> None:
        """
        Update the occupancy grid from a depth map, marking free and occupied space.

        Parameters:
            depth_map : The depth map of shape (H, W)
            intrinsics: Camera intrinsic matrix of shape (3, 3)
            extrinsics: Camera extrinsic matrix of shape (4, 4). world-to-camera
            surface_dist_thre: threshold that free space has to be away from occupied voxel
        """
        # Get all unexplored voxel coordinates
        D, H, W = self.occupancy_grid.shape
        x, y, z = torch.meshgrid(torch.arange(D, device=self.device), torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        grid_indices = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        unexplored_mask = self.occupancy_grid.flatten() == 0
        grid_indices = grid_indices[unexplored_mask]
        
        # Convert grid indices to world coordinates
        world_coords_sim = self.origin + grid_indices * self.voxel_size
        
        # Transform world coordinates to camera coordinates
        N = world_coords_sim.shape[0]
        homogeneous_coords_sim = torch.cat((world_coords_sim, torch.ones(N, 1, device=self.device)), dim=1)
        homogeneous_coords_slam = torch.mm(homogeneous_coords_sim, self.sim2slam.T)
        camera_coords = torch.mm(homogeneous_coords_slam, extrinsics.T)[:, :3] # from world_slam to camera_slam

        # Project camera coordinates to pixel coordinates
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        u = (camera_coords[:, 0] * fx / camera_coords[:, 2]) + cx
        v = (camera_coords[:, 1] * fy / camera_coords[:, 2]) + cy
        depth = camera_coords[:, 2]

        # Filter valid projections within image bounds
        H, W = depth_map.shape
        valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth > 0)
        u = u[valid_mask].long()
        v = v[valid_mask].long()
        depth = depth[valid_mask]
        grid_indices = grid_indices[valid_mask]

        # Update occupancy grid based on depth map
        depth_map_values = depth_map[v, u]
        occupied_mask = (depth_map_values - depth).abs() < self.voxel_size

        free_mask = (depth_map_values - depth) > surface_dist_thre  # becoz SplaTAM doesn't update when the camera is close to a surface, keep free region away 5 voxels from the surface
        free_indices, neglected_free_indices = self.find_free_indices(self.occupancy_grid, grid_indices[free_mask], dist_thre=surface_dist_thre, batch_size=find_free_indices_bs)

        # Mark free voxels
        # new_free_mask = self.occupancy_grid[free_indices[:, 0], free_indices[:, 1], free_indices[:, 2]] != -1
        # self._new_free_voxels = free_indices[new_free_mask]
        self.occupancy_grid[free_indices[:, 0], free_indices[:, 1], free_indices[:, 2]] = -1.0
        self.occupancy_grid[neglected_free_indices[:, 0], neglected_free_indices[:, 1], neglected_free_indices[:, 2]] = -2.0

        # Mark occupied voxels
        occupied_indices = grid_indices[occupied_mask]
        self.occupancy_grid[occupied_indices[:, 0], occupied_indices[:, 1], occupied_indices[:, 2]] = 1.0
    
    def update_prev_free_voxels(self, use_xyz_filter: bool = True, xy_sampling_step: float = 1.0, gs_z_levels = None):
        """ update last stored free grid
        
        Attributes:
            prev_free_voxles: [N, 3]. indices of free voxels
            
        """
        self.prev_free_voxels = self.get_free_voxels(use_xyz_filter, xy_sampling_step, gs_z_levels)

    def get_new_free_voxels(self, use_xyz_filter: bool = True, xy_sampling_step: float = 1.0, gs_z_levels = None) -> torch.Tensor:
        """ get free voxels compared to last stored free grid

        Args:
            use_xyz_filter: use XYZ location filter
    
        Returns:
            new_free_voxels: [N, 3]. indices of free voxels
        """
        prev_free_voxels = self.prev_free_voxels
        new_free_voxels = self.get_free_voxels(use_xyz_filter, xy_sampling_step, gs_z_levels)

        # Flatten each row into a single unique value (row-wise hashing)
        prev_flat = prev_free_voxels.view(-1, 1, 3)
        new_flat = new_free_voxels.view(1, -1, 3)
        
        # Find elements in new_free_voxels that don't match any in prev_free_voxels
        mask = (prev_flat == new_flat).all(dim=-1).any(dim=0)
        unique_new_voxels = new_free_voxels[~mask]

        return unique_new_voxels
        # return self._new_free_voxels
    
    def get_free_voxels(self, use_xyz_filter: bool = True, xy_sampling_step: float = 1.0, gs_z_levels = None) -> torch.Tensor:
        """ get free voxels in the global map
        
        Args:
            use_xyz_filter: use XYZ location filter
            xy_sampling_step: XY sampling step unit(meter)

        Returns:
            free_voxels: [N, 3]. indices of free voxels
        """
        free_mask = self.occupancy_grid == -1.0
        free_voxels = torch.stack(torch.where(free_mask), dim=1)

        if use_xyz_filter:
            gs_z_levels = torch.tensor(gs_z_levels, dtype=free_voxels.dtype, device=free_voxels.device)
            num_skip_vxl = xy_sampling_step / self.voxel_size # voxel_size = 0.05
            xyz_mask = (free_voxels[:, 0] % num_skip_vxl == 0) * (free_voxels[:, 1] % num_skip_vxl == 0) * (free_voxels[:, 2:3] == gs_z_levels).any(dim=1)
            filetered_free_voxels = free_voxels[xyz_mask]
            return filetered_free_voxels
        else:
            return free_voxels

    @staticmethod
    def transform_world_to_camera(world_coords: torch.Tensor, camera_extrinsics: torch.Tensor):
        """
        Transform occupancy grid coordinates from world to camera coordinates.

        Parameters:
            world_coords     : Coordinates in the world frame, shape (N, 3)
            camera_extrinsics: Extrinsic matrix [R|t] of shape (4, 4)

        Returns:
            camera_coords (torch.Tensor): Transformed coordinates in the camera frame, shape (N, 3)
        """
        # Convert world_coords to homogeneous coordinates (N, 4)
        N = world_coords.shape[0]
        homogeneous_coords = torch.cat((world_coords, torch.ones(N, 1, device=world_coords.device)), dim=1)
        
        # Apply the extrinsic transformation to obtain camera coordinates
        camera_coords = torch.mm(homogeneous_coords, camera_extrinsics[:3, :].T)
        
        return camera_coords[:, :3]

    def visualize(self, time_idx: int = 0, in_slam_world: bool = False):
        """
        Visualize the exploration map using Open3D. (in the SLAM coordinate system)

        Parameters:
            time_idx     : current iteration
            in_slam_world: convert point cloud to be in SplaTAM coorindate system, otherwise in simulation/GT system
        """
        import open3d as o3d

        # Get world coordinates of occupied, free, and unexplored voxels
        occupied_coords = self.get_world_coordinates_from_grid(1.0, in_slam_world).cpu().numpy()
        free_coords = self.get_world_coordinates_from_grid(-1.0, in_slam_world).cpu().numpy()
        unexplored_coords = self.get_world_coordinates_from_grid(0.0, in_slam_world).cpu().numpy()

        # Create point clouds for each type of voxel
        occupied_pcd = o3d.geometry.PointCloud()
        occupied_pcd.points = o3d.utility.Vector3dVector(occupied_coords)
        occupied_pcd.paint_uniform_color([1, 0, 0])  # Red for occupied

        free_pcd = o3d.geometry.PointCloud()
        free_pcd.points = o3d.utility.Vector3dVector(free_coords)
        free_pcd.paint_uniform_color([0, 1, 0])  # Green for free space

        unexplored_pcd = o3d.geometry.PointCloud()
        unexplored_pcd.points = o3d.utility.Vector3dVector(unexplored_coords)
        unexplored_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for unexplored

        # Mark the 8 corners of the bounding box with different colors
        corners = np.array([
            [self.bounding_box[0][0], self.bounding_box[1][0], self.bounding_box[2][0]],
            [self.bounding_box[0][0], self.bounding_box[1][0], self.bounding_box[2][1]],
            [self.bounding_box[0][0], self.bounding_box[1][1], self.bounding_box[2][0]],
            [self.bounding_box[0][0], self.bounding_box[1][1], self.bounding_box[2][1]],
            [self.bounding_box[0][1], self.bounding_box[1][0], self.bounding_box[2][0]],
            [self.bounding_box[0][1], self.bounding_box[1][0], self.bounding_box[2][1]],
            [self.bounding_box[0][1], self.bounding_box[1][1], self.bounding_box[2][0]],
            [self.bounding_box[0][1], self.bounding_box[1][1], self.bounding_box[2][1]]
        ])
        corner_colors = [
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
            [1, 1, 0],  # Yellow
            [0.5, 0.5, 0],  # Olive
            [0.5, 0, 0.5],  # Purple
            [0, 0.5, 0.5],  # Teal
            [0.5, 0.5, 1],  # Light Blue
            [1, 0.5, 0.5]  # Light Red
        ]
        corner_pcds = []
        for i, corner in enumerate(corners):
            corner_pcd = o3d.geometry.PointCloud()
            if in_slam_world:
                corner = (self.sim2slam.cpu().numpy() @  np.concatenate([corner, np.array([1.0])]))[:3]
            corner_pcd.points = o3d.utility.Vector3dVector([corner])
            corner_pcd = corner_pcd.voxel_down_sample(voxel_size=0.1)
            corner_pcd.points = o3d.utility.Vector3dVector([corner])
            corner_pcd.paint_uniform_color(corner_colors[i])
            corner_pcds.append(corner_pcd)

        # Store the colored point cloud as a PLY file
        all_pcds = [occupied_pcd] + corner_pcds
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in all_pcds:
            combined_pcd += pcd
        o3d.io.write_point_cloud(f'tmp/3d_plot/combined_{time_idx:04}.ply', combined_pcd)
        # o3d.io.write_point_cloud(f'tmp/3d_plot/occupied_{time_idx:04}.ply', occupied_pcd)
        # o3d.io.write_point_cloud(f'tmp/3d_plot/free_{time_idx:04}.ply', free_pcd)
        # o3d.io.write_point_cloud(f'tmp/3d_plot/unexplored_{time_idx:04}.ply', unexplored_pcd)
