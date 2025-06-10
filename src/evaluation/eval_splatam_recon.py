import argparse
import numpy as np
import os
from scipy.spatial import cKDTree as KDTree
import sys
import torch
import trimesh
from typing import Dict
from plyfile import PlyData, PlyElement
from matplotlib import cm
from tqdm import tqdm 

sys.path.append(os.getcwd())
from src.utils.general_utils import update_results_file


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def voxel_grid_sampling(points, voxel_size):
    """
    Perform voxel grid sampling on a set of points.

    Args:
        points (np.ndarray): Input points array of shape (N, 3).
        voxel_size (float): The size of each voxel.

    Returns:
        np.ndarray: Indices of points selected as representatives of each voxel.
        dict: A dictionary mapping voxel indices to the indices of points in that voxel.
    """
    # Quantize the points into voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)
    voxel_dict = {}

    for idx, voxel in tqdm(enumerate(voxel_indices), desc="Grouping points into voxels"):
        # Use a tuple of voxel indices as the key for the dictionary
        # This allows us to group points that fall into the same voxel
        # Note: Using a tuple as a key allows for easy grouping of points in the same voxel
        voxel_key = tuple(voxel)
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = []
        voxel_dict[voxel_key].append(idx)

    # Select the closest point to the voxel center from each voxel
    candidates = []
    for voxel_key, point_indices in tqdm(voxel_dict.items()):
        voxel_center = np.array(voxel_key) * voxel_size + voxel_size / 2
        closest_point = min(point_indices, key=lambda idx: np.linalg.norm(points[idx] - voxel_center))
        candidates.append(closest_point)

    return np.array(candidates), voxel_dict


def evenly_sample_points(gt_pts, pred_pts, pred_colors, num_sample_pts, voxel_size):
    """
    Evenly sample points from two point clouds using voxel grid sampling.

    Args:
        gt_pts (np.ndarray): Ground truth points array of shape (N, 3).
        pred_pts (np.ndarray): Predicted points array of shape (M, 3).
        pred_colors (np.ndarray): Predicted colors of the points array of shape (M, 3).
        num_sample_pts (int): Number of points to sample.
        voxel_size (float): The size of each voxel for sampling.

    Returns:
        np.ndarray: Final selected ground truth points.
        np.ndarray: Corresponding predicted points nearest to the selected ground truth points.
        np.ndarray: Corresponding predicted colors 
    """
    # Perform voxel grid sampling on gt_pts
    candidate_indices, voxel_dict = voxel_grid_sampling(gt_pts, voxel_size)
    
    # Randomly choose num_sample_pts from candidates
    if len(candidate_indices) > num_sample_pts:
        chosen_indices = np.random.choice(candidate_indices, size=num_sample_pts, replace=False)
    else:
        chosen_indices = candidate_indices

    # Get corresponding voxel coordinates for the selected points
    chosen_voxel_coords = np.floor(gt_pts[chosen_indices] / voxel_size).astype(int)

    # Find nearest points in pred_pts to the chosen voxel coordinates
    voxel_centers = chosen_voxel_coords * voxel_size + voxel_size / 2
    pred_tree = KDTree(pred_pts)
    _, nearest_pred_indices = pred_tree.query(voxel_centers, k=1)

    return gt_pts[chosen_indices], pred_pts[nearest_pred_indices], pred_colors[nearest_pred_indices]


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio, distances < dist_th


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc, distances


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp, distances


def color_pointcloud_by_values(pointcloud: trimesh.PointCloud, values: np.ndarray, output_file: str = "colored_pointcloud.ply"):
    """
    Color a trimesh.PointCloud based on a set of values, with red as highest and blue as lowest, then save as PLY.

    Args:
        pointcloud (trimesh.PointCloud): The point cloud with vertices shape (N, 3).
        values (np.ndarray): Array of values (N,) to map to colors.
        output_file (str): The path to save the colored point cloud as a PLY file.
    """
    # Normalize the values between 0 and 1
    norm_values = (values - values.min()) / (values.max() - values.min())

    # Map normalized values to colors (using a color map, e.g., coolwarm)
    colormap = cm.get_cmap('coolwarm')
    colors = (colormap(norm_values)[:, :3] * 255).astype(np.uint8)  # RGB, scaled to 0-255

    # Set the colors to the point cloud
    pointcloud.colors = colors

    # Save the colored point cloud as a PLY file
    pointcloud.export(output_file)
    print(f"Colored point cloud saved as {output_file}")


def save_filtered_pointcloud(pointcloud: trimesh.PointCloud, mask: np.ndarray, output_file: str = "filtered_pointcloud.ply"):
    """
    Filter and save a point cloud with only points that have values below a given threshold.

    Args:
        pointcloud (trimesh.PointCloud): The original point cloud with vertices shape (N, 3).
        mask (np.ndarray): Array of values (N,). 
        threshold (float): Threshold for filtering values.
        output_file (str): Path to save the filtered point cloud as a PLY file.
    """
    # Identify points with values below the threshold
    # mask = values > threshold
    filtered_vertices = pointcloud.vertices[~mask]
    
    # If colors exist, filter them as well
    if pointcloud.colors is not None:
        filtered_colors = pointcloud.colors[~mask]
        filtered_pointcloud = trimesh.PointCloud(filtered_vertices, colors=filtered_colors)
    else:
        filtered_pointcloud = trimesh.PointCloud(filtered_vertices)
    
    # Save the filtered point cloud
    filtered_pointcloud.export(output_file)
    print(f"Filtered point cloud saved as {output_file}")


def argument_parsing() -> argparse.Namespace:
    """parse arguments

    Returns:
        args: arguments

    """
    parser = argparse.ArgumentParser(
        description="Arguments to run NVS evaluation."
    )
    parser.add_argument("--ckpt", type=str, default=None,
                        help="ckpt to load in")
    parser.add_argument("--gt_mesh", type=str,
                        help="ground truth mesh file path")
    parser.add_argument("--transform_traj", type=str,
                        help="trajectory file for initial pose loading")
    parser.add_argument("--result_dir", type=str, help="result dir")
    args = parser.parse_args()
    return args

def load_scene_params(scene_path: str) -> Dict:
    """ load SplaTAM checkpoint parameters
    """
    params = dict(np.load(scene_path, allow_pickle=True))
    return params


def save_pointcloud_ply(xyz: np.ndarray, rgb: np.ndarray, filename: str) -> None:
    """
    Saves a colored point cloud to a PLY file.

    Args:
        xyz (np.ndarray): Nx3 array of point cloud coordinates.
        rgb (np.ndarray): Nx3 array of color values corresponding to each point.
        filename (str): The name of the output PLY file.

    Returns:
        None
    """
    if xyz.shape != rgb.shape or xyz.shape[1] != 3:
        raise ValueError("xyz and rgb arrays must have the same shape and must be Nx3.")
    
    ### Combine xyz and rgb arrays into a structured array for PLY
    points = np.zeros(len(xyz), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    ### Assign xyz and rgb to the structured array
    points['x'], points['y'], points['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    points['red'], points['green'], points['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    ### Create a PlyElement from the structured array
    el = PlyElement.describe(points, 'vertex')
    
    ### Write the data to a PLY file
    PlyData([el], text=True).write(filename)

    print(f"Point cloud saved as {filename}")


def sample_point_cloud(point_cloud: np.ndarray, max_points: int) -> np.ndarray:
    """
    Uniformly samples points from a point cloud, ensuring that the total number of points
    does not exceed max_points.

    Args:
        point_cloud (np.ndarray): A point cloud of shape (N, 3).
        colors (np.ndarray): A color point cloud of shape (N, 3).
        max_points (int): The maximum number of points to sample.

    Returns:
        np.ndarray: The sampled point cloud with shape (min(N, max_points), 3).
    """
    # Check that point cloud has shape (N, 3)
    assert point_cloud.shape[1] == 3, "Point cloud must be of shape (N, 3)."
    
    num_points = point_cloud.shape[0]
    
    # If number of points is less than or equal to max_points, return the point cloud as is
    if num_points <= max_points:
        return point_cloud
    
    # Randomly sample indices from the point cloud without replacement
    sampled_indices = np.random.choice(num_points, size=max_points, replace=False)
    return sampled_indices
    

def convert_gs_to_pcd(
        params: Dict,
        ply_path: str = None
        ) -> np.ndarray:
    """ Sample point clouds from GS. Save PLY if needed

    Args:
        params: SplaTAM parameters
        ply_path: Pointcloud PLY path

    """
    xyz = params['means3D']
    color_precomp = params['rgb_colors']
    sample_idxs, _ = voxel_grid_sampling(xyz, 0.01)
    xyz = xyz[sample_idxs]
    color_precomp = color_precomp[sample_idxs]

    color_precomp = (color_precomp * 255).astype(np.uint8)
    if ply_path:
        save_pointcloud_ply(xyz, color_precomp, ply_path)
    return xyz, color_precomp


def calc_3d_pcd_metric(rec_pc_tri, gt_pc_tri, return_pt_err=False):
    """
    3D reconstruction metric.

    """
    accuracy_rec, dist_acc = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec, dist_comp = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec, comp_mask = completion_ratio(
        gt_pc_tri.vertices, rec_pc_tri.vertices)
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %

    if return_pt_err:
        pt_err = {'dist_acc': dist_acc, 'dist_comp': dist_comp, 'comp_mask': comp_mask}
        results = {'acc': accuracy_rec, 'comp': completion_rec, 'comp%': completion_ratio_rec}
        return results, pt_err
    else:
        return {'acc': accuracy_rec, 'comp': completion_rec, 'comp%': completion_ratio_rec}


def load_Replica_pose(line: str) -> torch.Tensor:
    """ load Replica pose from trajectory file

    Args:
        line (str): pose data as txt line. Format: camera-to-world, RUB

    Returns:
        c2w (torch.Tensor, [4,4]): pose. Format: camera-to-world, RDF
    """
    c2w = np.array(list(map(float, line.split()))).reshape(4, 4) 
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    c2w = torch.from_numpy(c2w).float()
    return c2w


def transform_point_cloud(transformation: torch.Tensor, point_cloud: np.ndarray) -> np.ndarray:
    """
    Transforms a point cloud using a 4x4 transformation matrix.

    Args:
        transformation (torch.Tensor): A 4x4 camera-to-world transformation matrix.
        point_cloud (np.ndarray): A point cloud of shape (N, 3).

    Returns:
        np.ndarray: The transformed point cloud of shape (N, 3).
    """
    # Ensure transformation is a 4x4 tensor
    assert transformation.shape == (4, 4), "Transformation matrix must be 4x4."
    assert point_cloud.shape[1] == 3, "Point cloud must be of shape (N, 3)."
    
    # Convert the point cloud to homogeneous coordinates (N, 4)
    num_points = point_cloud.shape[0]
    homogeneous_points = np.ones((num_points, 4))  ### comment: Create a (N, 4) array with ones in the last column
    homogeneous_points[:, :3] = point_cloud  ### comment: Assign the original 3D coordinates to the first three columns
    
    # Apply the transformation (convert transformation to numpy for matmul if necessary)
    transformation_np = transformation.cpu().numpy() if isinstance(transformation, torch.Tensor) else transformation
    transformed_points = homogeneous_points @ transformation_np.T  ### comment: Matrix multiplication (N, 4) x (4, 4)^T
    
    # Convert back from homogeneous to 3D by dropping the fourth column
    transformed_points = transformed_points[:, :3] / transformed_points[:, 3, np.newaxis]  ### comment: Normalize by the last coordinate if it's not 1
    
    return transformed_points


args = argument_parsing()

##################################################
### load GT
##################################################
mesh_gt = trimesh.load(args.gt_mesh, process=False)
if args.gt_mesh.endswith('obj'):
    mesh_gt = as_mesh(mesh_gt)
gt_pc = trimesh.sample.sample_surface_even(mesh_gt, 200000)[0]

##################################################
### Load prediction
##################################################
params = load_scene_params(args.ckpt)
xyz, colors = convert_gs_to_pcd(params)

## transform point cloud to GT world coordinate system ##
if args.transform_traj:
    traj_txt = os.path.join(args.transform_traj)
    with open(traj_txt, 'r') as f:
        lines = f.readlines()
        poses = [load_Replica_pose(line) for line in lines]

slam2sim = poses[0] # RDF
slam2sim[:3, 1] *= -1 
slam2sim[:3, 2] *= -1 # RUB
pred_pc = transform_point_cloud(slam2sim, xyz)


##################################################
### sample gt and predictions
### convert them to trimesh.PointCloud
##################################################
gt_pc, pred_pc, colors = evenly_sample_points(gt_pc, pred_pc, colors, 200000, 0.01)
gt_pc_tri = trimesh.PointCloud(vertices=gt_pc)

rec_pc_tri = trimesh.PointCloud(vertices=pred_pc)
rec_pc_tri.colors = colors

##################################################
### evaluate
##################################################
eval_result, pt_err = calc_3d_pcd_metric(rec_pc_tri, gt_pc_tri, True)

##################################################
### print and save result
##################################################
print(eval_result)
os.makedirs(args.result_dir, exist_ok=True)
result_txt = os.path.join(args.result_dir, "eval_3d_result.txt")
update_results_file(eval_result, result_txt)

### save point cloud ###
# gt_pc_tri.export(os.path.join(args.result_dir, "gt_pc_tri.ply"))
rec_pc_tri.export(os.path.join(args.result_dir, "rec_pc_tri.ply"))
color_pointcloud_by_values(rec_pc_tri, pt_err['dist_acc'], os.path.join(args.result_dir, "pt_err_accuracy.ply"))
color_pointcloud_by_values(gt_pc_tri, pt_err['dist_comp'], os.path.join(args.result_dir, "pt_err_completeness.ply"))
save_filtered_pointcloud(gt_pc_tri, pt_err['comp_mask'],  os.path.join(args.result_dir, "pt_err_complete_ratio.ply"))
