from glob import glob
import shutil
import gc
import torch
import json
import torch.nn as nn
import cv2
import torch.nn.functional as F
from time import strftime
import os, sys, time
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import platform
import scipy
import numpy as np
from PIL import Image
import re

from src.utils.preprocess_fromvideo import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.facerender.pirender_animate import AnimateFromCoeff_PIRender
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from torchvision.transforms import PILToTensor
from src.utils.visualize import save_landmarks_to_video
from src.appearance_guided_landmark_matching import Landmark_Matching, get_angle_and_distance, Appearance_Matching
from src.dift.models.inv_dift_sd import SDFeaturizer
from src.retargeting_tools.tools import *

from pytorch_lightning import seed_everything
from torchvision import transforms

seed_everything(42, workers=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def main(args):
    save_dir = args.result_dir
    os.makedirs(save_dir, exist_ok=True)
    device = args.device
    driving_pose = args.driving_pose
    image_file = args.source_image

    crop = transforms.Compose([
        transforms.Resize(args.size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.size),
    ])

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size,
                                args.old_version, args.preprocess)
    # init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    # extract landmarks from driving video (human)

    driving_pose_videoname = os.path.splitext(os.path.split(driving_pose)[-1])[0]
    driving_pose_frame_dir = os.path.join(save_dir, driving_pose_videoname)
    os.makedirs(driving_pose_frame_dir, exist_ok=True)
    print('3DMM Extraction for the driving video providing pose')
    driving_lms = preprocess_model.generate_lms(driving_pose, driving_pose_frame_dir, pic_size=args.size,
                                                num_frames=args.num_frames)

    os.makedirs(save_dir, exist_ok=True)
    save_landmarks_to_video(driving_lms, f"{save_dir}/driving_landmarks.mp4")

    # facial parts
    part_index = [list(range(0, 17)), list(range(48, 55)), list(range(54, 60)) + [48], list(range(31, 36)),
                  list(range(60, 65)), list(range(64, 68)) + [60],
                  list(range(36, 40)), list(range(39, 42)) + [36],
                  list(range(42, 46)), list(range(45, 48)) + [42],
                  list(range(17, 22)), list(range(22, 27)),
                  list(range(27, 31)), ]

    # return origin and theta for each frame in driving video
    driving_infos = processing_params(part_index, driving_lms)

    # appearance-guided landmark matching
    dift = SDFeaturizer(inversion=False)
    reference_image = Image.open(image_file).convert('RGB')

    prompt = 'a photo of a face'
    img_size = args.size
    reference_image = crop(reference_image)
    print("reference_image size:", reference_image.size)

    reference_img_tensor = transform(reference_image)
    reference_feature = dift.forward(reference_img_tensor.to(device), reference_image, prompt=prompt).cpu()
    print('Processing landmark matching ...')
    
    target_feature = Appearance_Matching(image_file, img_size)
    
    ref_lm = Landmark_Matching(f"{save_dir}/reference.png", reference_image, reference_feature, target_feature, img_size)

    ref_lm = np.expand_dims(ref_lm, axis=0)
    ref_infos = processing_params(part_index, ref_lm)

    ref_lms = np.repeat(ref_lm, driving_lms.shape[0], axis=0)

    # boundary b_ref and b_dri
    driving_boundary = process_boundary(driving_lms, driving_infos, part_index)
    ref_boundary = process_boundary(ref_lms, ref_infos, part_index)

    # boundary scale b_ref/b_dri
    boundary_scale = process_boundary_scale(driving_boundary, ref_boundary)
    boundary_scale = [[min(value, 1) for value in sublist] for sublist in boundary_scale]

    driving_infos[3]['params'] = [driving_infos[-1]['params'][i][:2] + driving_infos[3]['params'][i][2:] for i in
                                  range(len(driving_infos[3]['params']))]
    ref_infos[3]['params'] = [ref_infos[-1]['params'][i][:2] + ref_infos[3]['params'][i][2:] for i in
                              range(len(ref_infos[3]['params']))]

    # global discrepancies of ogirin and theta
    global_offset = [{'delta_origin': np.array([driving_infos[0]['params'][idx + 1][0], driving_infos[0]['params'][idx + 1][1]])
                      - np.array([driving_infos[0]['params'][0][0], driving_infos[0]['params'][0][1]]), 'delta_theta': driving_infos[0]['params'][idx + 1][4] 
                      - driving_infos[0]['params'][0][4], } for idx in range(len(driving_lms[1:]))]

    for i in range(len(part_index)):
        ref_part_param, driving_part_param = ref_infos[i]['params'][0], driving_infos[i]['params'][0]
        tmp_lms = []

        # normailize points for the first frame
        def process_normalize(lm, param):
            lm_norm = normalize_points(param[4], lm, param[0], param[1])
            lm_norm[(np.abs(lm_norm) > 0) & (np.abs(lm_norm) < 1)] = 0
            return lm_norm

        driving_part_lm_norm = process_normalize(driving_lms[0, part_index[i], :], driving_part_param)
        ref_part_lm_norm = process_normalize(ref_lm[0, part_index[i], :], ref_part_param)

        # retargeting
        for idx, driving_part_lm_next in enumerate(driving_lms[1:, part_index[i], :]):
            driving_part_param_next = driving_infos[i]['params'][idx + 1]
            delta_origin = np.array([driving_part_param_next[0], driving_part_param_next[1]]) - np.array([driving_part_param[0], driving_part_param[1]])
            driving_part_lm_next_norm = process_normalize(driving_part_lm_next, driving_part_param_next)
            
            ref_part_lm_next_norm, scale, auxiliary_point, tmp_driving_part_lm_norm = (
                np.zeros_like(ref_part_lm_norm) for _ in range(4)
            )

            for c in range(len(ref_part_lm_norm)):
                # auxiliary_point if driving points equal 0
                auxiliary_point[c] = np.where(ref_part_lm_norm[c] == 0, driving_part_lm_norm[c], 0)

                for b in range(ref_part_lm_next_norm.shape[1]):
                    if driving_part_lm_next_norm[c][b] == 0:
                        scale[c][b] = 0
                        continue

                    driving_zero, ref_zero = driving_part_lm_norm[c][b] == 0, ref_part_lm_norm[c][b] == 0
                    
                    # constrain scale to a reasonable value
                    threshold = 1.5
                    ref_greater = np.abs(ref_part_lm_norm[c][b] / driving_part_lm_norm[c][b]) > threshold if not driving_zero else False
                    if driving_zero:
                        tmp_zero = np.abs(driving_part_lm_norm[c][(b + 1) % 2] / ref_part_lm_norm[c][(b + 1) % 2]) if (driving_part_lm_norm[c][(b + 1) % 2] != 0 and ref_part_lm_norm[c][(b + 1) % 2] != 0) else 1
                        tmp_driving_part_lm_norm[c][b] = tmp_zero if ref_zero else np.abs(ref_part_lm_norm[c][b])
                        number = (0 if np.abs(driving_part_lm_next_norm[c][b]) > np.abs(tmp_driving_part_lm_norm[c][b] * threshold) else 1)
                        scale[c][b] = driving_part_lm_next_norm[c][b] / tmp_driving_part_lm_norm[c][b] if ref_zero else (number + np.abs(driving_part_lm_next_norm[c][b] / tmp_driving_part_lm_norm[c][b])) * (driving_part_lm_next_norm[c][b] *ref_part_lm_norm[c][b] / np.abs(driving_part_lm_next_norm[c][b] * ref_part_lm_norm[c][b]))

                    else:
                        tmp_driving_part_lm_norm[c][b] = np.abs(ref_part_lm_norm[c][b]) * (driving_part_lm_norm[c][b] / np.abs(driving_part_lm_norm[c][b])) if ref_greater else driving_part_lm_norm[c][b]

                        number = (0 if np.abs(driving_part_lm_next_norm[c][b] - driving_part_lm_norm[c][b]) > np.abs(tmp_driving_part_lm_norm[c][b] * threshold) else 1)
                        if (driving_part_lm_next_norm[c][b] - driving_part_lm_norm[c][b]) / tmp_driving_part_lm_norm[c][b] < 0 and number == 1:
                            number = -1
                        scale[c][b] = number + (driving_part_lm_next_norm[c][b] - driving_part_lm_norm[c][b]) / tmp_driving_part_lm_norm[c][b] if (np.abs(driving_part_lm_next_norm[c][b] / driving_part_lm_norm[c][b]) > 1 and ref_greater) else driving_part_lm_next_norm[c][b] / driving_part_lm_norm[c][b]

            post_driving_part_lm_norm = driving_part_lm_norm.copy()
            post_driving_part_lm_next_norm = driving_part_lm_next_norm.copy()
            
            # handling of extreme points
            if (i == 4 or i == 5) and scale[2][1] == 0 and (scale[1][1] != 0 and scale[3][1] != 0):
                scale[2][1] = (scale[1][1] + scale[3][1]) / 2.0
                post_driving_part_lm_norm[2][1] = (post_driving_part_lm_norm[1][1] + post_driving_part_lm_norm[3][1]) / 2.0
                post_driving_part_lm_next_norm[2][1] = (driving_part_lm_next_norm[1][1] + driving_part_lm_next_norm[3][1]) / 2.0
                    
            # constrain scale to a reasonable value
            if i <= 9:
                scale = smooth_scale(scale)
            
            # local point motion
            for c in range(len(ref_part_lm_norm)):
                ref_part_lm_next_norm[c][0] = (ref_part_lm_norm[c][0] if auxiliary_point[c][0] == 0 else auxiliary_point[c][0]) * scale[c][0] - (auxiliary_point[c][0] if scale[c][0] != 0 else 0) + (scale[c][0] if auxiliary_point[c][0] == 0 and ref_part_lm_norm[c][0] == 0 else 0)
                ref_part_lm_next_norm[c][1] = (ref_part_lm_norm[c][1] if auxiliary_point[c][1] == 0 else auxiliary_point[c][1]) * scale[c][1] - (auxiliary_point[c][1] if scale[c][1] != 0 else 0) + (scale[c][1] if auxiliary_point[c][1] == 0 and ref_part_lm_norm[c][1] == 0 else 0)
            
            # handling of extreme points
            ref_part_lm_next_norm = post_process(post_driving_part_lm_norm, post_driving_part_lm_next_norm,ref_part_lm_norm,ref_part_lm_next_norm)

            ref_part_lm_next = denormalize_points(ref_part_param[4], ref_part_lm_next_norm, ref_part_param[0],ref_part_param[1])
            
            # global motion
            offset_ref_part_lm_next = offset_transfer(ref_part_lm_next, ref_part_param[0], ref_part_param[1], ref_infos[0]['params'][0][4], driving_infos[0]['params'][idx + 1][4],
                                                      global_offset[idx]['delta_theta'], global_offset[idx]['delta_origin'], [1.0, 1.0, 1.0, 1.0])
            # local relative motion
            if i != 0:
                offset_ref_part_lm_next = offset_transfer(offset_ref_part_lm_next, (offset_ref_part_lm_next[0, 0] + offset_ref_part_lm_next[-1, 0]) / 2, 
                                                          (offset_ref_part_lm_next[0, 1] + offset_ref_part_lm_next[-1, 1]) / 2, ref_infos[i]['params'][0][4] + global_offset[idx]['delta_theta'], 
                                                          driving_infos[i]['params'][idx + 1][4], driving_infos[i]['params'][idx + 1][4] - global_offset[idx]['delta_theta'] - driving_infos[i]['params'][0][4], 
                                                          delta_origin - global_offset[idx]['delta_origin'], boundary_scale[i - 1])

            tmp_lms.append(offset_ref_part_lm_next)

        tmp_lms = np.array(tmp_lms)
        ref_lms[1:, part_index[i], :] = tmp_lms

    save_landmarks_to_video(np.array(ref_lms), f"{save_dir}/reference_landmarks.mp4")

    landmark_path = f"{save_dir}/reference_landmarks.npy"
    driving_path = f"{save_dir}/driving_landmarks.npy"
    np.save(landmark_path, ref_lms)
    np.save(driving_path, driving_lms)
    return landmark_path, driving_path


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("--driven_audio", default='./sadtalker_video2pose/dummy/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--driving_pose", default=None, help="path to driving video providing pose")
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--checkpoint_dir", default='./ckpts/sad_talker', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--size", type=int, default=512, help="the image size of the facerender")
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'],
                        help="how to preprocess the images")
    parser.add_argument("--old_version", action="store_true", help="use the pth other than safetensor version")
    parser.add_argument("--source_image", required=True, help="path for source image")
    args = parser.parse_args()

    args.device = "cuda"

    main(args)
