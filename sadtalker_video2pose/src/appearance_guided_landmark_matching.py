import json
from scipy.ndimage import gaussian_filter
from PIL import Image
from skimage.measure import EllipseModel, CircleModel
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import gc
import os


def get_angle_and_distance(x, y, xc, yc, a, b, theta):
    cos_theta = np.cos(-theta)
    sin_theta = np.sin(-theta)
    
    x_rot = (x - xc) * cos_theta - (y - yc) * sin_theta
    y_rot = (x - xc) * sin_theta + (y - yc) * cos_theta

    if b == 0:
        angle = np.arctan2(0, x_rot / a)
        distance = np.abs(y_rot)
    else:
        angle = np.arctan2(y_rot / b, x_rot / a)

        x_on_ellipse = a * np.cos(angle)
        y_on_ellipse = b * np.sin(angle)

        distance = np.sqrt((x_rot - x_on_ellipse) ** 2 + (y_rot - y_on_ellipse) ** 2)

        if (x_rot ** 2 / a ** 2 + y_rot ** 2 / b ** 2) < 1.0:
            distance = -distance

    return angle, distance


def get_index(index, offset, n, out_indices):
    while index in out_indices:
        index = (index + offset) % n
    return index


def line_transfer(point):
    coeff = np.polyfit(point[[0, -1], 0], point[[0, -1], 1], 1)
    line_func = np.poly1d(coeff)

    x_start, x_end = point[0, 0], point[-1, 0]

    x_new = np.linspace(x_start, x_end, point.shape[0])
    y_new = line_func(x_new)
    uniform_points = np.vstack((x_new, y_new)).T

    adjusted_points = []
    for i, (x, y) in enumerate(point):
        y_line = line_func(x)
        distance = (y - y_line) / np.sqrt(1 + coeff[0] ** 2)
        angle = np.arctan(coeff[0])  
        x_adjusted = uniform_points[i, 0] - distance * np.sin(angle)
        y_adjusted = uniform_points[i, 1] + distance * np.cos(angle)
        adjusted_points.append([x_adjusted, y_adjusted])

    adjusted_points = np.array(adjusted_points)

    return adjusted_points[1:-1]


def get_angle_and_distance_circle(x, y, xc, yc, r):
    angle = np.arctan2(y - yc, x - xc)

    distance = np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r
    return angle, distance


def ellipse_transfer(point):
    try:
        ellipse = EllipseModel()
        ellipse.estimate(point)
        ellipse_flag = ellipse.params is not None
    except:
        ellipse_flag = False
    try:
        circle = CircleModel()
        circle.estimate(point)
        circle_flag = circle.params is not None
    except:
        circle_flag = False

    if ellipse_flag and len(point) > 15:
        xc, yc, a, b, theta = ellipse.params
        angles_distances = np.array([get_angle_and_distance(x, y, xc, yc, a, b, theta) for x, y in point])
        angles = angles_distances[:, 0]
        distances = angles_distances[:, 1]

        angles = np.unwrap(angles)

        t = np.linspace(0, 1, len(angles)) * (angles[-1] - angles[0]) + angles[0]

        x_fit = xc + (a + distances[1:-1]) * np.cos(t[1:-1]) * np.cos(theta) - (b + distances[1:-1]) * np.sin(t[1:-1]) * np.sin(theta)
        y_fit = yc + (a + distances[1:-1]) * np.cos(t[1:-1]) * np.sin(theta) + (b + distances[1:-1]) * np.sin(t[1:-1]) * np.cos(theta)
        point[1:-1] = np.column_stack((x_fit, y_fit))
    elif circle_flag:
        xc, yc, r = circle.params
        angles_distances = np.array([get_angle_and_distance_circle(x, y, xc, yc, r) for x, y in point])
        angles = angles_distances[:, 0]
        distances = angles_distances[:, 1]
        angles = np.unwrap(angles)

        t = np.linspace(0, 1, len(angles)) * (angles[-1] - angles[0]) + angles[0]

        x_fit = xc + (r + distances[1:-1]) * np.cos(t[1:-1])
        y_fit = yc + (r + distances[1:-1]) * np.sin(t[1:-1])

        point[1:-1] = np.column_stack((x_fit, y_fit))
    else:
        point[1:-1] = line_transfer(point)
    return np.round(point).astype(int)


def nose_transfer(point):
    fixed_index = -1
    fixed_point = point[fixed_index]

    x_offsets = point[:, 0] - fixed_point[0]
    y_offsets = point[:, 1] - fixed_point[1]

    remaining_x_offsets = np.delete(x_offsets, fixed_index)
    remaining_y_offsets = np.delete(y_offsets, fixed_index)

    slope = np.sum(remaining_x_offsets * remaining_y_offsets) / np.sum(remaining_x_offsets ** 2)

    x_new = np.linspace(point[0, 0], fixed_point[0], point.shape[0])
    y_new = slope * (x_new - fixed_point[0]) + fixed_point[1]
    uniform_points = np.vstack((x_new, y_new)).T

    return uniform_points

# rearrangement based on the facial prior knowledge for better visualization
def rearrangement(points):
    ellipse_names = [list(range(0,17)), list(range(17,22)), list(range(22,27)), list(range(31,36)),
                     list(range(36,40)), list(range(39,42)) + [36],
                     list(range(42, 46)), list(range(45, 48)) + [42],
                     list(range(48, 55)), list(range(54, 60)) + [48],
                     list(range(60, 65)), list(range(64, 68)) + [60],]
    for name in ellipse_names:
        point = points[name]
        point = point[:, [1, 0]]

        uniform_points = ellipse_transfer(point)

        points[name] = uniform_points[:, [1, 0]]

    nose_name = list(range(27,31)) + [33]
    points[nose_name] = nose_transfer(points[nose_name])
    return points


def Appearance_Matching(img_path, img_size):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    
    prompts = {"face_boundary": "完整框出图中脸的下半部分,要求只包含半个鼻子,请确保你的输出只有一个box", "eye_brows": "完整框出图中眉毛的位置,请确保你的输出只有一个box", "nose": "完整框出图中鼻子的位置,请确保你的输出只有一个box", "eyes": "用一个框完整框出图中两只眼睛的位置,请确保你的输出只有一个box", "mouth": "完整框出图中嘴巴的位置,请确保你的输出只有一个box"}
    
    image_encoder_path = "ckpts/ip-adapter/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(
        "cuda", dtype=torch.float32
    )
    
    clip_image_processor = CLIPImageProcessor()
    logit_scale = 100
    
    target_domains = "target_domains"
    
    target_features = []
    
    for part, prompt in prompts.items():
        query = tokenizer.from_list_format([
            {'image': img_path},  # Either a local path or an url
            {'text': prompt},
        ])
        response, history = model.chat(tokenizer, query=query, history=None)
        x1, y1, x2, y2 = tokenizer._fetch_all_box_with_ref(response)[0]['box']
        x1, y1, x2, y2 = (int(x1 / 1000 * img_size), int(y1 / 1000 * img_size), int(x2 / 1000 * img_size), int(y2 / 1000 * img_size))
        image = Image.open(img_path).crop((x1, y1, x2, y2))
        clip_image = clip_image_processor(images=image, return_tensors="pt").pixel_values
        clip_image_embeds = image_encoder(clip_image.to("cuda", dtype=torch.float32)).image_embeds
        
        clip_image_embeds = clip_image_embeds / clip_image_embeds.norm(dim=1, keepdim=True).to(torch.float32)
        
        domain_path = os.path.join(target_domains, part)
        domains = sorted(os.listdir(domain_path))
        
        name, value = "", 0
        for domain in domains:
            domain_clip_feature = torch.load(os.path.join(domain_path, domain, "clip_feature.pt"))
            domain_clip_feature = domain_clip_feature / domain_clip_feature.norm(dim=1, keepdim=True).to(torch.float32)
            clip_score = logit_scale * (clip_image_embeds * domain_clip_feature).sum()
            
            print(f"facial part {part}, domain {domain}, clip_score {clip_score}")
            
            if clip_score > value:
                name = domain
                value = clip_score

        target_features.append(torch.load(os.path.join(domain_path, name, "diff_feature.pt")))
  
    return torch.cat(target_features)
        

def Landmark_Matching(name, imgs, ft, tar_ft, img_size, scatter_size=10):
    num_imgs = len(ft)
    fig, axes = plt.subplots(1, num_imgs, figsize=(5 * num_imgs, 5))
    plt.tight_layout()
    num_channel = ft.size(1)
    axes.imshow(imgs)
    axes.axis('off')
    axes.set_title('Matching Result')
    
    matching_points = []
    trg_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(ft.to("cuda"))  # N, C, H, W
    trg_vec = trg_ft.view(1, num_channel, -1)  # N, C, HW
    trg_vec = F.normalize(trg_vec)  # N, C, HW
    for idx, src_vec in enumerate(tar_ft):
        src_vec = src_vec.unsqueeze(0)

        cos_map = torch.matmul(src_vec.to("cuda"), trg_vec).view(1, img_size, img_size).cpu().numpy().squeeze(0)  # N, H, W
        max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)

        matching_points.append(max_yx)

    del trg_ft, trg_vec, src_vec
    gc.collect()
    torch.cuda.empty_cache()
    
    matching_points = np.array(matching_points)
    matching_points = rearrangement(matching_points)

    for max_yx in matching_points:
        axes.scatter(max_yx[1].item(), max_yx[0].item(), c='r', s=scatter_size)

    del cos_map
    # del heatmap
    gc.collect()
    plt.savefig(name)
    plt.close()
    return matching_points[:, [1, 0]]