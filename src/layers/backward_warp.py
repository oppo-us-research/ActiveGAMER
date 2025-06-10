import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numba as nb
from typing import Tuple

from src.layers.backprojection import Backprojection
from src.layers.projection import Projection
from src.layers.transformation3d import Transformation3D

from time import time

TIMING_PRINT = True


def sort_tensor(tensor: torch.Tensor, sort_ref: torch.Tensor) -> torch.Tensor:
    """sort the tensor in ascending order according to the value in the last dimension

    Args:
        tensor (torch.Tensor, [N, C]): tensor to be sorted
        sort_ref (torch.Tensor, [N]): reference for sorting

    Returns:
        sorted_tensor (torch.Tensor, [N, C]): sorted tensor
    """
    # Get the last value of each element
    # last_col = tensor[:, 5]

    # Sort the tensor along the last column
    sorted_tensor, indices = torch.sort(sort_ref)

    # Index the tensor using the sorted indices
    sorted_tensor = tensor[indices]
    return sorted_tensor


def find_first_occurance(indexing_tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor `indexing_tensor`, return the first occurance of unique elements.

    Args:
        indexing_tensor (torch.Tensor, [N]): A 1-D tensor to find first occurance of unique elements.

    Returns:
        first_occ_index (torch.Tensor): The first occurance of unique elements in the input tensor.

    Example:
    >>> indexing_tensor = torch.tensor([1, 2, 3, 2, 1])
    >>> find_first_occurance(indexing_tensor)
    tensor([0, 1, 2])
    """
    start = time()
    unique, idx, counts = torch.unique(indexing_tensor, sorted=True, return_inverse=True, return_counts=True)
    # breakpoint()
    # print("---- unique: ", (time()-start)*1000)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((cum_sum[0:1] * 0., cum_sum[:-1])).long()
    first_indicies = ind_sorted[cum_sum]
    return first_indicies


class BackwardWarping(nn.Module):
    """ Layer to forward (depth) warping a perspective image
    """

    def __init__(self,
                 out_hw: Tuple[int, int],
                 device: torch.device,
                 K: torch.Tensor
                 ) -> None:
        """
        Args:
            out_hw (Tuple[int, int]): output image size
            device (torch.device): torch device
            K (torch.Tensor, [4,4]): camera intrinsics
        """
        super(BackwardWarping, self).__init__()
        height, width = out_hw
        self.backproj = Backprojection(height, width).to(device)
        self.projection = Projection(height, width).to(device)
        self.transform3d = Transformation3D().to(device)

        H, W = height, width
        self.rgb = torch.zeros(H, W, 3).view(-1, 3).to(device)
        self.depth = torch.zeros(H, W, 1).view(-1, 1).to(device)
        self.K = K.to(device)
        self.inv_K = torch.inverse(K).to(device)
        self.K = self.K.unsqueeze(0)
        self.inv_K = self.inv_K.unsqueeze(0)

    # def create_rgbdm(self,
    #                  inputs: torch.Tensor, # uvrgbd, h, w
    #                  out_hw: Tuple[int, int],
    #                  ) -> torch.Tensor:
    #     """ create RGB-Depth-Mask with z-buffer

    #     Args:
    #         inputs (torch.Tensor, [N,D]): input data for creating rgbdm. uvrgbd(m)
    #         out_hw (Tuple): output image size

    #     Returns:
    #         rgbdm (torch.Tensor, [N,D, H,W]): RGB-Depth-Mask, where mask==1 indicates missing pixels
    #     """
    #     ### FIXME: upgrade this for [B,N,6] input ###
    #     H, W = out_hw

    #     ### FIXME: debug ###
    #     # times = {}
    #     # start = time()

    #     ### prepare data ###
    #     ### FIXME: debug ###
    #     # times['-- [FW_Torch] prepare data'] = time() - start
    #     # start = time()

    #     ### rm invalid depth points ###
    #     valid_depth_mask = (inputs[-1] < 1e6) & (inputs[-1] > 0) # H,W,

    #     ### rm invalid coords ###
    #     vaild_coord_mask = (inputs[0] > -1) & (inputs[0] < 1) & (inputs[1] > -1) & (inputs[1] < 1)

    #     ### compute vaild mask ###
    #     valid_mask = valid_depth_mask & vaild_coord_mask

    #     ### initialize output data ###
    #     rgb_inputs = inputs[2:-1,:].unsqueeze(0) # N,C,H,W
    #     depth_inputs = inputs[-1:,:].unsqueeze(0) # N,1,H,W
    #     vgrids = inputs[:2, :].clone().unsqueeze(0).permute(0, 2, 3, 1) # N,H,W,2

    #     rgb = torch.nn.functional.grid_sample(rgb_inputs, vgrids, mode='nearest', padding_mode='border')  # N.C,H,W
    #     depth = torch.nn.functional.grid_sample(depth_inputs, vgrids, mode='bilinear', padding_mode='border') # N.1,H,W

    #     ### reshape as image ###
    #     # rgb = rgb.permute(0, 2, 3, 1)  # N,H,W,C
    #     # depth = depth.permute(0, 2, 3, 1) # N,H,W,1

    #     ### create mask ###
    #     valid_mask = valid_mask.view(-1, 1, out_hw[0],out_hw[1]) # N,1,H,W
    #     mask = ~valid_mask

    #     rgbdm = torch.cat([rgb, depth, mask], dim=1)  # N,D,H,W

    #     ### FIXME: debug ###
    #     # for key, val in times.items():
    #     #     print(f"{key}: {val*1000:.02f}ms")
    #     # print("-"*20)
    #     return rgbdm.float()

    # def backward_warping(self,
    #                     img: torch.Tensor,
    #                     depth: torch.Tensor,
    #                     coord: torch.Tensor,  # N,H,W,2
    #                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """

    #     Args:
    #         img (torch.Tensor, [N, 3, H, W]): image
    #         depth (torch.Tensor, [N, 1, H, W]): depth
    #         coord (torch.Tensor, [N, H, W, 2]): coordinate for backward warping

    #     Returns:
    #         new_rgb (torch.Tensor, [N, 3, H, W]): warped image
    #         new_depth (torch.Tensor, [N, 1, H, W]): warped depth in the tgt coordinates, need forward project to ref coordinate
    #         new_mask (torch.Tensor, [N, 1, H, W]): mask for w/ and w/o valid coordinates

    #     """
    #     ### prepare data ###
    #     N, _, H, W = depth.shape
    #     uv = coord.permute(0, 3, 1, 2) # NCHW
    #     backward_input = torch.cat([uv, img, depth], dim=1)  # N,uvrgbd,h,w
    #     input_dim = backward_input.shape[1]
    #     # backward_input = backward_input.reshape(N, input_dim, H, W)  # N, M, D

    #     ### create RGB-D-M per sample ###
    #     rgbdm_all = []
    #     for i in range(N):
    #         rgbdm = self.create_rgbdm(backward_input[i], (H, W))
    #         rgbdm_all.append(rgbdm)   # 1,D,H,W

    #     ### concatenate result ###
    #     rgbdm_all = torch.cat(rgbdm_all)  # N, H, W, rgbdmask
    #     # rgbdm_all = rgbdm_all.permute(0, 3, 1, 2)
    #     new_rgb, new_depth, new_mask = rgbdm_all[:, :-2], rgbdm_all[:, -2:-1], rgbdm_all[:, -1:]

    #     return new_rgb, new_depth, new_mask

    # ## Backward warp depth map D2 (refer view) to candidate view (D1), X(x,y,z) is world coordinate, u(u,v) is 2d coordinate

    def forward(self,
                img_tgt: torch.Tensor, # D
                depth_tgt: torch.Tensor,
                depth_ref: torch.Tensor,
                T: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            img (torch.Tensor, [N,C,H,W]: color map
            depth (torch.Tensor, [N,1,H,W]: depth map
            T (torch.Tensor, [N,4,4]): reference-to-target (nv)

        Returns:
            nv_img (torch.Tensor, [N,C,H,W]): novel view image
            nv_depth (torch.Tensor, [N,1,H,W]): novel view depth
            nv_mask (torch.Tensor, [N,1,H,W]): novel view mask
        """
        ### get shape ###
        b, _, h, w = img_tgt.shape

        ### reprojection ###
        pts3d = self.backproj(depth_ref, self.inv_K)  # [N,4,(HxW)] (x,y,z,1) X1@1
        pts3d_nv = self.transform3d(pts3d, T)   # X1@2 depth1@2 = pts3d_nv[:,2]
        nv_grid = self.projection(pts3d_nv, self.K, normalized=True)  # [N,H,W,2] u1@2 = K2 \dot X1@2
        transformed_distance = pts3d_nv[:, 2:3].view(b, 1, h, w)  # N, 1, H, W   depth1@2 = pts3d_nv[:,2]

        ### backward warping ###
        # nv_img, nv_depth_warp, nv_mask = self.backward_warping(img_tgt, depth_tgt, nv_grid)
        nv_img = F.grid_sample(img_tgt, nv_grid)
        nv_depth = F.grid_sample(depth_tgt, nv_grid)

        ### FIXME: debug ###
        # times['-- [FW_Torch] backward_warping'] = time() - start
        # start = time()
        # if need the warp depth in the ref coordinate system, need to do forward project based on the sample depth

        # pt3d_trans = self.backproj(nv_depth_warp, self.inv_K)
        # pt3d_ref =  self.transform3d(pts3d, torch.linalg.inv(T))
        # nv_depth = pt3d_ref[:,2].view(-1, 1, h,w)


        # return nv_img.float(), nv_depth_warp.float(), nv_mask.float(), transformed_distance.float()
        return nv_img, nv_depth, transformed_distance


if __name__ == "__main__":
    ### packages ###
    import cv2
    import os
    import matplotlib.pyplot as plt
    from time import time

    ##################################################
    ### Arguments
    ##################################################
    data_dir = "data/dibr_test_data/replica_sim_mvs/room_0/"
    depth_dir = "data/dibr_test_data/replica_sim_mvs/room_0/pinhole_depth_000"
    device = "cuda"
    out_hw = (640, 640)

    ##################################################
    ### Initialize Layer
    ##################################################
    K = torch.eye(4)
    K[0, 0] = out_hw[1] / 2
    K[0, 2] = out_hw[1] / 2
    K[1, 1] = out_hw[0] / 2
    K[1, 2] = out_hw[0] / 2
    K = K.to(device).float()

    backward_warp = BackwardWarping(
        out_hw=out_hw,
        device=device,
        K=K
    )

    ##################################################
    ### Load Data
    ##################################################
    ref_i = 0
    tgt_i = 10

    ### RGB ###
    img_path = os.path.join(data_dir, "pinhole_color_000", f"{ref_i:06}.png")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (out_hw[1], out_hw[0]))
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float()

    img_tgt_path = os.path.join(data_dir, "pinhole_color_000", f"{tgt_i:06}.png")
    img_tgt = cv2.imread(img_tgt_path)
    img_tgt = cv2.cvtColor(img_tgt, cv2.COLOR_BGR2RGB)
    img_tgt = cv2.resize(img_tgt, (out_hw[1], out_hw[0]))
    img_tgt = torch.from_numpy(img_tgt).permute(2, 0, 1).unsqueeze(0).to(device).float()

    ### depth ###
    depth_path = os.path.join(data_dir, "pinhole_depth_000", f"{ref_i:06}.png")
    depth = cv2.imread(depth_path, -1) / 6553.5  # H,W
    depth = cv2.resize(depth, (out_hw[1], out_hw[0]))
    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device).float()

    depth_tgt_path = os.path.join(data_dir, "pinhole_depth_000", f"{tgt_i:06}.png")
    depth_tgt = cv2.imread(depth_tgt_path, -1) / 6553.5  # H,W
    depth_tgt = cv2.resize(depth_tgt, (out_hw[1], out_hw[0]))
    depth_tgt = torch.from_numpy(depth_tgt).unsqueeze(0).unsqueeze(0).to(device).float()

    ### relative pose ###
    pose_txt = os.path.join(data_dir, "pinhole_color_000_pose", f"{ref_i:06}.txt")
    c2w_ref = np.loadtxt(pose_txt)  # 4,4
    pose_txt = os.path.join(data_dir, "pinhole_color_000_pose", f"{tgt_i:06}.txt")
    c2w_tgt = np.loadtxt(pose_txt)  # 4,4
    ref2tgt = np.linalg.inv(c2w_tgt) @ c2w_ref
    ref2tgt = torch.from_numpy(ref2tgt).to(device).unsqueeze(0).float()  # 1, 4,4

    ##################################################
    ### Backward warping
    ##################################################
    start = time()
    # nv_img, nv_depth, nv_mask, depth_trans = backward_warp(img, depth_tgt, depth, ref2tgt)
    nv_img, nv_depth, depth_trans = backward_warp(img_tgt, depth_tgt, depth, ref2tgt)
    print("==> BackwardWarp: {}ms".format(
        (time() - start) * 1000
    ))
    nv_img_np = nv_img[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    # nv_mask_np = np.clip(nv_mask[0, 0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
    nv_depth_np = nv_depth[0, 0].detach().cpu().numpy().astype(np.float32)
    nv_depth_mean = nv_depth_np[nv_depth_np < 1e6].mean()
    nv_depth_np[nv_depth_np > 1e6] = nv_depth_mean

    ##################################################
    ### Inpainting
    ##################################################
    # inpaint_method = cv2.INPAINT_NS
    # bs = 0  # border size to be excluded (usually large patch)
    # if bs > 0:
    #     nv_img_inpaint = cv2.inpaint(nv_img_np[bs:-bs, bs:-bs], nv_mask_np[bs:-bs, bs:-bs], 3, inpaint_method)
    #     start = time()
    #     nv_depth_inpaint = cv2.inpaint(nv_depth_np[bs:-bs, bs:-bs], nv_mask_np[bs:-bs, bs:-bs], 3, inpaint_method)
    #     print("==> Inpaint: {}ms".format(
    #         (time() - start) * 1000
    #     ))
    # else:
    #     nv_img_inpaint = cv2.inpaint(nv_img_np, nv_mask_np, 3, inpaint_method)
    #     start = time()
    #     nv_depth_inpaint = cv2.inpaint(nv_depth_np, nv_mask_np, 3, inpaint_method)
    #     print("==> Inpaint: {}ms".format(
    #         (time() - start) * 1000
    #     ))

    ##################################################
    ### Plot result
    ##################################################
    f, ax = plt.subplots(2, 4)
    vis = img[0].permute(1, 2, 0).detach().cpu().numpy() / 255
    ax[0, 0].imshow(vis)

    vis = depth[0, 0].detach().cpu().numpy()
    ax[1, 0].imshow(vis, vmin=0, vmax=10)

    depth_img_path = os.path.join(data_dir, "pinhole_color_000", f"{tgt_i:06}.png")
    depth_img = cv2.imread(depth_img_path)
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
    ax[0, 1].imshow(depth_img)

    depth_path = os.path.join(data_dir, "pinhole_depth_000", f"{tgt_i:06}.png")
    # depth_path = os.path.join(depth_dir, f"room_0_pinhole_color_000_{tgt_i:06}.png")
    depth = cv2.imread(depth_path, -1) / 6553.5  # H,W
    ax[1, 1].imshow(depth, vmin=0, vmax=10)

    vis = nv_img_np
    ax[0, 2].imshow(vis)

    vis = nv_depth[0, 0].detach().cpu().numpy()
    ax[1, 2].imshow(vis, vmin=0, vmax=10)

    img_ref = img[0].permute(1, 2, 0).detach().cpu().numpy()
    warp_img = nv_img[0].permute(1, 2, 0).detach().cpu().numpy()
    img_diff = np.mean(np.abs(img_ref - warp_img), axis=2)
    ax[0, 3].imshow(img_diff)

    depth_trans = depth_trans[0, 0].detach().cpu().numpy()
    nv_depth = nv_depth[0, 0].detach().cpu().numpy()
    depth_diff = np.abs(depth_trans - nv_depth)
    ax[1, 3].imshow(depth_diff, vmin=0, vmax=2)

    plt.show()