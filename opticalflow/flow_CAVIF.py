import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import imageio
import matplotlib.pyplot as plt
import sys
import matplotlib.cm as cm

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from model_selection import model_type, pre_trained_model_types, select_model
from test_models import pad_to_same_shape

torch.set_grad_enabled(False)
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from utils_flow.visualization_utils import overlay_semantic_mask, make_sparse_matching_plot
from utils_flow.util_optical_flow import flow_to_image
from models.inference_utils import estimate_mask
from utils_flow.flow_and_mapping_operations import convert_flow_to_mapping
from validation.utils import matches_from_flow
from admin.stats import DotDict
import cv2


def getFlowAndPR(network, source_image, target_image, threshold=0.5):
    query_image = source_image
    reference_image = target_image

    query_image_shape = query_image.shape
    ref_image_shape = reference_image.shape

    estimated_flow, uncertainty_components = network.estimate_flow_and_confidence_map(query_image, reference_image)

    uncertainty_key = 'p_r'  # 'inv_cyclic_consistency_error'
    # 'p_r', 'inv_cyclic_consistency_error' can also be used as a confidence measure
    # 'cyclic_consistency_error' can also be used, but that's an uncertainty measure
    confidence_map = uncertainty_components[uncertainty_key]
    # confidence_map = confidence_map[:, :, :ref_image_shape[0], :ref_image_shape[1]]

    # zwyking codes here:
    flow_est = estimated_flow
    p_r = confidence_map
    W = flow_est.shape[3]
    H = flow_est.shape[2]
    B = flow_est.shape[0]
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    grid = grid.to(flow_est.device)
    flow_warp = grid + flow_est
    flow_x = flow_warp[:, 0, ...].unsqueeze(1)
    flow_y = flow_warp[:, 1, ...].unsqueeze(1)
    flow_x_flag = torch.logical_and(flow_x > 0, flow_x < W)
    flow_y_flag = torch.logical_and(flow_y > 0, flow_y < H)
    flow_flag = torch.logical_and(flow_x_flag, flow_y_flag)
    flow_flag = torch.logical_and(p_r > threshold, flow_flag)
    p_r[flow_flag] = 1.0
    p_r[~flow_flag] = 0.0
    p_r = p_r.squeeze(1)

    return flow_est, p_r


def frame_count_to_patch_info(frame_count, patch_gap, sequence_len):
    patch_id = frame_count // (patch_gap * sequence_len) * patch_gap + (frame_count % patch_gap)
    index_in_patch = (frame_count % (patch_gap * sequence_len)) // patch_gap
    return patch_id, index_in_patch


# choose model
model = 'PDCNet_plus'
pre_trained_model = 'megadepth'
flipping_condition = False
global_optim_iter = 3
local_optim_iter = 7
# path_to_pre_trained_models = '/cluster/work/cvl/truongp/DenseMatching/pre_trained_models/'
path_to_pre_trained_models = "./pre_trained_models/PDCNet_plus_megadepth.pth.tar"

# inference parameters for PDC-Net
network_type = model  # will only use these arguments if the network_type is 'PDCNet' or 'PDCNet_plus'
choices_for_multi_stage_types = ['d', 'h', 'ms']
multi_stage_type = 'd'
if multi_stage_type not in choices_for_multi_stage_types:
    raise ValueError('The inference mode that you chose is not valid: {}'.format(multi_stage_type))

confidence_map_R = 1.0
ransac_thresh = 1.0
mask_type = 'proba_interval_1_above_10'  # for internal homo estimation
homography_visibility_mask = True
scaling_factors = [0.5, 0.6, 0.88, 1, 1.33, 1.66, 2]
compute_cyclic_consistency_error = True  # here to compare multiple uncertainty

# usually from argparse
args = DotDict(
    {'network_type': network_type, 'multi_stage_type': multi_stage_type, 'confidence_map_R': confidence_map_R,
     'ransac_thresh': ransac_thresh, 'mask_type': mask_type,
     'homography_visibility_mask': homography_visibility_mask, 'scaling_factors': scaling_factors,
     'compute_cyclic_consistency_error': compute_cyclic_consistency_error})

# define network and load network weights
network, estimate_uncertainty = select_model(
    model, pre_trained_model, args, global_optim_iter, local_optim_iter,
    path_to_pre_trained_models=path_to_pre_trained_models)
estimate_uncertainty = True

SHOW_MASK = True
videoFolder = "videos/inputs/CUST_unstable"
save_path = "videos/outputs"
export_path = "videos/export"
videos = [""]
resize = (1920, 1080)

for video in videos:
    if video.endswith(".mp4"):
        videoName = video[:-4]
        print("processing:", video)

        # 创建输出目录
        optical_dir = os.path.join(export_path, videoName, "optical_flow")
        conf_dir = os.path.join(export_path, videoName, "pdc_confidence")
        os.makedirs(optical_dir, exist_ok=True)
        os.makedirs(conf_dir, exist_ok=True)

        vid = cv2.VideoCapture(os.path.join(videoFolder, video))
        width = resize[0]  # int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = resize[1]  # int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        totalNFrame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        # 2024.1.3 zwyking's code here:
        video_patch = {}
        rval = True
        patch_gap = 2
        sequence_len = 5
        frame_count = 0
        conf_valid = 0.5
        save_data = {}
        print("----------start store frame:----------")

        while rval:
            patch_id = frame_count // (patch_gap * sequence_len) * patch_gap + (frame_count % patch_gap)
            # print(frame_count, "PATCH ID:", patch_id)
            rval, frame = vid.read()
            if frame is not None:
                frame = cv2.resize(frame, resize)
            if rval:
                if patch_id not in video_patch.keys():
                    video_patch[patch_id] = []
                if str(patch_id) not in save_data.keys():
                    save_data[str(patch_id)] = {}
                    save_data[str(patch_id)]["flow_map"] = []
                    save_data[str(patch_id)]["conf_map"] = []

                video_patch[patch_id].append(frame)
                frame_count += 1

        print("FRAMES: ", frame_count)

        print("--------start processing ---------")

        with torch.no_grad():
            for id in range(len(video_patch)):
                input_images = torch.from_numpy(np.stack(video_patch[id])).permute(0, 3, 1, 2)
                source_img = input_images[1:, ...]
                source_img_original = torch.cat([source_img, input_images[0, ...].unsqueeze(0)], dim=0)
                target_img_original = input_images
                # print(source_img_original.shape, target_img_original.shape)
                # exit()
                flow_est, p_r = getFlowAndPR(network, source_img_original, target_img_original, threshold=conf_valid)
                # print(flow_est.shape, p_r.shape) # torch.Size([5, 2, 1080, 1920]) torch.Size([5, 1080, 1920])

                # save flow_map
                flow_est = flow_est.permute(0, 2, 3, 1).detach().cpu().numpy()
                # save_data[str(id)]["flow_map"].append(flow_est)

                # save conf_map
                p_r = p_r.detach().cpu().numpy()
                p_r_save = p_r.copy()
                for idx in range(p_r.shape[0] - 1, 0, -1):
                    pr_fron = p_r[idx]
                    flow = flow_est[idx - 1, ...]
                    warp_pr_fron = remap_using_flow_fields(pr_fron, flow[..., 0], flow[..., 1]) > conf_valid
                    pr_back = p_r[idx - 1] > conf_valid
                    pr_new = np.logical_and(pr_back, warp_pr_fron)
                    p_r_save[idx - 1] = pr_new
                    p_r[idx - 1] = warp_pr_fron
                save_data[str(id)]["conf_map"].append(p_r_save)
                # print(flow_est.shape, p_r_save.shape) #(5, 1080, 1920, 2) (5, 1080, 1920)
                print(id + 1, "/", len(video_patch))

        print("---------start export result--------")
        if SHOW_MASK:
            out = cv2.VideoWriter(os.path.join(save_path, videoName + "_conf_test.mp4"),
                                  cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        for cur_frame in range(frame_count - 1):  # frame_count
            cur_patch, index_in_patch = frame_count_to_patch_info(cur_frame, patch_gap, sequence_len)
            next_patch, next_index_in_patch = frame_count_to_patch_info(cur_frame + 1, patch_gap, sequence_len)

            # print("frame {} ID {} index {}".format(cur_frame, cur_patch, index_in_patch))
            frame = video_patch[cur_patch][index_in_patch]
            frame_next = video_patch[next_patch][next_index_in_patch]
            conf = save_data[str(cur_patch)]["conf_map"][0][index_in_patch]
            # flow = save_data[str(cur_patch)]["flow_map"][0][index_in_patch]
            # print(frame.shape, conf.shape) #(1080, 1920, 3) (1080, 1920)

            flow, _ = getFlowAndPR(network, \
                                   torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0), \
                                   torch.from_numpy(frame_next).permute(2, 0, 1).unsqueeze(0), threshold=conf_valid)
            # print(flow.shape)
            flow = flow[0]

            # ---- 保存为 .npy 文件 ----
            # 光流: [2, H, W] -> [H, W, 2]
            flow_np = flow.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)  # [H, W, 2]
            conf_np = conf.astype(np.float32)  # [H, W]

            np.save(os.path.join(optical_dir, f"frame_{cur_frame:06d}.npy"), flow_np)
            np.save(os.path.join(conf_dir, f"frame_{cur_frame:06d}.npy"), conf_np)

            # -- save visualize video
            if SHOW_MASK:
                flow_vis = flow_to_image(flow.detach().cpu().numpy())

                frame_vis = cv2.resize(frame, (width // 2, height // 2))
                conf_vis = np.uint8(conf[..., np.newaxis] * 255)
                conf_vis = cv2.resize(conf_vis, (width // 2, height // 2))
                conf_vis = cv2.cvtColor(conf_vis, cv2.COLOR_GRAY2BGR)
                flow_vis = cv2.resize(flow_vis, (width // 2, height // 2))

                row1 = np.concatenate((frame_vis, frame_vis), axis=1)
                row2 = np.concatenate((flow_vis, conf_vis), axis=1)
                result = np.concatenate((row1, row2), axis=0)
                result = np.uint8(result)
                # print(frame.shape, conf.shape, result.shape)
                out.write(result)
            print(cur_frame + 1, "/", frame_count - 1)

        vid.release()
        if SHOW_MASK:
            out.release()

        print(f"Saved to: {optical_dir} and {conf_dir}")

print("All videos processed!")