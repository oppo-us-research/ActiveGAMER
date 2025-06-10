import numpy as np
import torch
import torch.nn as nn
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
    cum_sum = torch.cat((cum_sum[0:1]*0., cum_sum[:-1])).long()
    first_indicies = ind_sorted[cum_sum]
    return first_indicies


class ForwardWarping(nn.Module):
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
        super(ForwardWarping, self).__init__()
        height, width = out_hw
        self.backproj = Backprojection(height, width).to(device)
        self.projection = Projection(height, width).to(device)
        self.transform3d = Transformation3D().to(device)

        H, W = height, width
        self.rgb = torch.zeros(H,W,3).view(-1,3).to(device)
        self.depth = torch.zeros(H,W,1).view(-1,1).to(device)
        self.K = K.to(device)
        self.inv_K = torch.inverse(K)
        self.K = self.K.unsqueeze(0)
        self.inv_K = self.inv_K.unsqueeze(0)

    def create_rgbdm(self,
                     inputs: torch.Tensor, 
                     out_hw: Tuple[int, int],
                     ) -> torch.Tensor:
        """ create RGB-Depth-Mask with z-buffer

        Args:
            inputs (torch.Tensor, [N,D]): input data for creating rgbdm. uvrgbd(m)
            out_hw (Tuple): output image size

        Returns:
            rgbdm (torch.Tensor, [H,W,5]): RGB-Depth-Mask, where mask==1 indicates missing pixels 
        """
        ### FIXME: upgrade this for [B,N,6] input ###
        H, W = out_hw
        
        ### FIXME: debug ###
        times = {}
        start = time()

        ### prepare data ###
        ### FIXME: debug ###
        times['-- [FW_Torch] prepare data'] = time() - start
        start = time()
        
        ### sort points according to depth ###
        inputs = sort_tensor(inputs, inputs[:, 5])

        ### FIXME: debug ###
        times['-- [FW_Torch] sort points'] = time() - start
        start = time()

        ### rm invalid depth points ###
        valid_mask = inputs[:, 5] < 1e6

        ### FIXME: debug ###
        times['-- [FW_Torch] find invalid'] = time() - start
        start = time()
        
        # print((valid_mask==0).sum())
        inputs = inputs[valid_mask]

        ### FIXME: debug ###
        times['-- [FW_Torch] keep valid'] = time() - start
        start = time()

        ### find unique ###
        inputs[:, :2] = torch.round(inputs[:, :2])   

        ### FIXME: debug ###
        times['-- [FW_Torch] find unique'] = time() - start
        start = time()
        
        ### clip ###
        inputs[:, 0] = torch.clip(inputs[:, 0], 0, W-1)
        inputs[:, 1] = torch.clip(inputs[:, 1], 0, H-1)

        ### FIXME: debug ###
        times['-- [FW_Torch] clipping'] = time() - start
        start = time()

        ### reindexing ###
        uv_1d = inputs[:, 1] * W + inputs[:, 0]
        uv_1d_unique_idx = find_first_occurance(uv_1d)

        ### FIXME: debug ###
        times['-- [FW_Torch] find first occ'] = time() - start
        start = time()

        ### get unique pts with min depth ###
        inputs = inputs[uv_1d_unique_idx]

        ### initialize output data ###
        rgb = self.rgb * 0.
        depth = self.depth * 0. + 1e8

        ### gather unique pts as images ###
        uv_1d_unique_idx2 = (inputs[:, 1] * W + inputs[:, 0]).long()
        rgb[uv_1d_unique_idx2] = inputs[:,2:5]
        depth[uv_1d_unique_idx2] = inputs[:,5:6]


        ### reshape as image ###
        rgb = rgb.reshape(H,W,3)
        depth = depth.reshape(H,W,1)

        ### FIXME: debug ###
        times['-- [FW_Torch] gather data'] = time() - start
        start = time()

        ### create mask ###
        mask = (depth == 0) * 1.

        rgbdm = torch.cat([rgb, depth, mask], dim=-1)

        ### inpaint nearby 4 corners ###
        rgbdm00 = rgbdm[:, W//4:W//4*3].clone()
        rgbdm01 = rgbdm[:, W//4:W//4*3].clone()
        rgbdm10 = rgbdm[:, W//4:W//4*3].clone()
        rgbdm11 = rgbdm[:, W//4:W//4*3].clone()

        rgbdm01[1:, :] = rgbdm00[:-1, :]
        rgbdm10[:, 1:] = rgbdm00[:, :-1]
        rgbdm11[1:, 1:] = rgbdm00[:-1, :-1]

        channel = rgbdm.shape[-1]
        depths = torch.stack([rgbdm00[:, :, 3], rgbdm01[:, :, 3], rgbdm10[:, :, 3], rgbdm11[:, :, 3]])
        reshaped_rgbdm = torch.stack([rgbdm00.reshape(-1, channel), rgbdm01.reshape(-1, channel), rgbdm10.reshape(-1, channel), rgbdm11.reshape(-1, channel)]) # 4,H*W,C
        reshaped_rgbdm = reshaped_rgbdm.permute(1,2,0)

        z_buffer_idx = torch.min(depths, dim=0).indices.reshape(-1)

        sorted_rgbdm = []

        times['-- [FW_Torch] get z buffer'] = time() - start
        start = time()

        ### copy channel datas ###
        for i in range(channel):
            selected_input = torch.gather(reshaped_rgbdm[:, i], 1, z_buffer_idx.view(-1,1))
            sorted_rgbdm.append(selected_input) # H*W,
        rgbdm_half = torch.stack(sorted_rgbdm, dim=1) # H*W, C
        rgbdm_half = rgbdm_half.reshape(H, W//2, -1)
        rgbdm[:, W//4:W//4*3] = rgbdm_half
        # rgbdm[:, :, 3][rgbdm[:, :, 3]>1e6] = 0

        times['-- [FW_Torch] copy channels'] = time() - start
        start = time()


        ### FIXME: debug ###
        # for key, val in times.items():
        #     print(f"{key}: {val*1000:.02f}ms")
        # print("-"*20)
        return rgbdm.float()
    
    def forward_warping(self, 
                        img: torch.Tensor, 
                        depth: torch.Tensor, 
                        coord: torch.Tensor,
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            img (torch.Tensor, [N, 3, H, W]): image
            depth (torch.Tensor, [N, 1, H, W]): depth
            coord (torch.Tensor, [N, H, W, 2]): coordinate for forward warping
    
        Returns:
            new_rgb (torch.Tensor, [N, 3, H, W]): warped image
            new_depth (torch.Tensor, [N, 1, H, W]): warped depth
            new_mask (torch.Tensor, [N, 1, H, W]): mask for w/ and w/o valid coordinates
    
        """
        ### prepare data ###
        N,_,H,W = depth.shape
        uv = coord.permute(0, 3, 1, 2)
        forward_input = torch.cat([uv, img, depth], dim=1) # uvrgbd
        input_dim = 6
        forward_input = forward_input.reshape(N, input_dim, -1).permute(0,2,1) # N, M, D

        ### create RGB-D-M per sample ###
        rgbdm_all = []
        for i in range(N):
            rgbdm = self.create_rgbdm(forward_input[i], (H, W))
            rgbdm_all.append(rgbdm)
        
        ### concatenate result ###
        rgbdm_all = torch.stack(rgbdm_all) # N, H, W, 5
        rgbdm_all = rgbdm_all.permute(0, 3, 1, 2)
        new_rgb, new_depth, new_mask = rgbdm_all[:, :3], rgbdm_all[:, 3:4], rgbdm_all[:, 4:5]
        
        return new_rgb, new_depth, new_mask

    def forward(self, 
                img: torch.Tensor, 
                depth: torch.Tensor, 
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
        b, _, h, w = img.shape

        ### FIXME: debug ###
        times = {}
        start = time()

        ### reprojection ###
        pts3d = self.backproj(depth, self.inv_K)

        ### FIXME: debug ###
        times['-- [FW_Torch] backproj'] = time() - start
        start = time()

        pts3d_nv = self.transform3d(pts3d, T)

        ### FIXME: debug ###
        times['-- [FW_Torch] transform3d'] = time() - start
        start = time()

        nv_grid = self.projection(pts3d_nv, self.K, normalized=False)
        ### FIXME: debug ###
        times['-- [FW_Torch] projection'] = time() - start
        start = time()

        ### create view-dependent mask ###
        ### get transformed depth ###
        transformed_distance = pts3d_nv[:, 2:3].view(b, 1, h, w) # N, 1, H, W

        ### forward warping ###
        nv_img, nv_depth, nv_mask = self.forward_warping(img, transformed_distance, nv_grid)

        ### FIXME: debug ###
        times['-- [FW_Torch] forward_warping'] = time() - start
        start = time()

        ### compute valid mask ###
        invalid_depth_mask = (nv_depth > 1e6) * 1. # maximum is 1e8
        nv_mask = ((nv_mask + invalid_depth_mask) > 0) * 1.

        ### FIXME: debug ###
        for key, val in times.items():
            print(f"{key}: {val*1000:.02f}ms")
        print("-"*20)

        return nv_img.float(), nv_depth.float(), nv_mask.float()


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
    out_hw = (320, 320)
    
    ##################################################
    ### Initialize Layer
    ##################################################
    K = torch.eye(4)
    K[0,0] = out_hw[1]/2
    K[0,2] = out_hw[1]/2
    K[1,1] = out_hw[0]/2
    K[1,2] = out_hw[0]/2
    K = K.to(device).float()

    forward_warp = ForwardWarping(
        out_hw = out_hw,
        device = device,
        K = K
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
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device).float()

    ### depth ###
    depth_path = os.path.join(data_dir, "pinhole_depth_000", f"{ref_i:06}.png")
    depth = cv2.imread(depth_path, -1) / 6553.5 # H,W
    depth = cv2.resize(depth, (out_hw[1], out_hw[0]))
    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device).float()

    ### relative pose ###
    pose_txt = os.path.join(data_dir, "pinhole_color_000_pose", f"{ref_i:06}.txt")
    c2w_ref = np.loadtxt(pose_txt) # 4,4
    pose_txt = os.path.join(data_dir, "pinhole_color_000_pose", f"{tgt_i:06}.txt")
    c2w_tgt = np.loadtxt(pose_txt) # 4,4
    ref2tgt = np.linalg.inv(c2w_tgt) @ c2w_ref
    ref2tgt = torch.from_numpy(ref2tgt).to(device).unsqueeze(0).float() # 1, 4,4

    ##################################################
    ### forward warping 
    ##################################################
    start = time()
    nv_img, nv_depth, nv_mask = forward_warp(img, depth, ref2tgt)
    print("==> ForwardWarp: {}ms".format(
        (time()-start) * 1000
    ))
    nv_img_np = nv_img[0].permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    nv_mask_np = np.clip(nv_mask[0,0].detach().cpu().numpy()*255, 0, 255).astype(np.uint8)
    nv_depth_np = nv_depth[0,0].detach().cpu().numpy().astype(np.float32)
    nv_depth_mean = nv_depth_np[nv_depth_np<1e6].mean()
    nv_depth_np[nv_depth_np>1e6] = nv_depth_mean
    
    ##################################################
    ### Inpainting
    ##################################################
    inpaint_method = cv2.INPAINT_NS
    bs = 0 # border size to be excluded (usually large patch)
    if bs > 0:
        nv_img_inpaint = cv2.inpaint(nv_img_np[bs:-bs, bs:-bs],nv_mask_np[bs:-bs, bs:-bs],3, inpaint_method)
        start = time()
        nv_depth_inpaint = cv2.inpaint(nv_depth_np[bs:-bs, bs:-bs],nv_mask_np[bs:-bs, bs:-bs],3, inpaint_method)
        print("==> Inpaint: {}ms".format(
            (time()-start) * 1000
        ))
    else:
        nv_img_inpaint = cv2.inpaint(nv_img_np ,nv_mask_np ,3, inpaint_method)
        start = time()
        nv_depth_inpaint = cv2.inpaint(nv_depth_np ,nv_mask_np ,3, inpaint_method)
        print("==> Inpaint: {}ms".format(
            (time()-start) * 1000
        ))
    
    ##################################################
    ### Plot result
    ##################################################
    f, ax = plt.subplots(2,4)
    vis = img[0].permute(1,2,0).detach().cpu().numpy()/255
    ax[0,0].imshow(vis)

    vis = depth[0,0].detach().cpu().numpy()
    ax[1,0].imshow(vis, vmin=0, vmax=10)


    img_path = os.path.join(data_dir, "pinhole_color_000", f"{tgt_i:06}.png")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax[0,1].imshow(img)

    depth_path = os.path.join(data_dir, "pinhole_depth_000", f"{tgt_i:06}.png")
    # depth_path = os.path.join(depth_dir, f"room_0_pinhole_color_000_{tgt_i:06}.png")
    depth = cv2.imread(depth_path, -1) / 6553.5 # H,W
    ax[1,1].imshow(depth, vmin=0, vmax=10)
    
    vis = nv_img_np
    ax[0,2].imshow(vis)

    vis = nv_depth[0,0].detach().cpu().numpy()
    ax[1,2].imshow(vis, vmin=0, vmax=10)

    
    ax[0,3].imshow(nv_img_inpaint)
    ax[1,3].imshow(nv_depth_inpaint, vmin=0, vmax=10)
    
    plt.show()
    