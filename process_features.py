import gc
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import PILToTensor
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
src_path = os.path.join(project_root, "sadtalker_video2pose")
sys.path.insert(0, src_path)
from sadtalker_video2pose.src.dift.models.inv_dift_sd import SDFeaturizer

seed_everything(42, workers=True)
transform = transforms.Compose([
    transforms.Resize((512,512), interpolation=transforms.InterpolationMode.BILINEAR), 
    transforms.CenterCrop(512)
])

def Inference(ft, img_size, positions):
    num_channel = ft[0].size(1)
    average_pos = []
    for pos, feat in zip(positions, ft):
        src_vecs = []
        src_ft = feat
        src_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(src_ft)
        for x, y in pos:
            with torch.no_grad():
                x, y = int(np.round(x)), int(np.round(y))
                src_vec = src_ft[0, :, y, x].view(1, num_channel)  # 1, C
                src_vecs.append(src_vec)
                del src_vec
                
        del src_ft
        gc.collect()
        torch.cuda.empty_cache()

        average_pos.append(torch.cat(src_vecs))
        del src_vecs
    average_pos = torch.stack(average_pos)
    average_pos = F.normalize(average_pos, p=2, dim=-1)
    average_pos = torch.mean(average_pos, dim=0)
    return average_pos


if __name__ == "__main__":
    torch.cuda.set_device(0)

    dift = SDFeaturizer(inversion=False)
    prompt = f'a photo of a face'
    img_size = 512
    ensemble_size = 8

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
    
    clip_features = {
        "face_boundary":[],
        "eye_brows":[],
        "nose":[],
        "eyes":[],
        "mouth":[],
    }  
    
    ref_positions = []
    ft = []
    
    image_dir = './characters/images/'
    points_dir = './characters/points/'
    character_domains = os.listdir(image_dir)
    
    target_dir = "target_domains"
    
    part_idx = {
        "face_boundary":[0,17],
        "eye_brows":[17, 27],
        "nose":[27,36],
        "eyes":[36,48],
        "mouth":[48,68],
    }

    for domain in character_domains:
        image_files = [f for f in os.listdir(os.path.join(image_dir, domain)) if f.endswith('.jpg')] 
        for image_file in image_files:
            image_path = os.path.join(image_dir, domain, image_file)
            for part, prompt in prompts.items():
                query = tokenizer.from_list_format([
                    {'image': image_path},  # Either a local path or an url
                    {'text': prompt},
                ])
                response, history = model.chat(tokenizer, query=query, history=None)
                x1, y1, x2, y2 = tokenizer._fetch_all_box_with_ref(response)[0]['box']
                x1, y1, x2, y2 = (int(x1 / 1000 * img_size), int(y1 / 1000 * img_size), int(x2 / 1000 * img_size), int(y2 / 1000 * img_size))
                crop_image = Image.open(image_path).crop((x1, y1, x2, y2))
                clip_image = clip_image_processor(images=crop_image, return_tensors="pt").pixel_values
                clip_image_embeds = image_encoder(clip_image.to("cuda", dtype=torch.float32)).image_embeds
                clip_image_embeds = clip_image_embeds / clip_image_embeds.norm(dim=1, keepdim=True).to(torch.float32)
                clip_features[part].append(clip_image_embeds)
            ref_img = Image.open(image_path).convert('RGB')
            positions = np.load(os.path.join(points_dir, domain, os.path.splitext(image_file)[0] + '.npy'))
            ref_positions.append(positions)
            ref_img = transform(ref_img)
            ref_img_tensor = (PILToTensor()(ref_img) / 255.0 - 0.5) * 2
            # diffution_ft
            ft.append(dift.forward(ref_img_tensor,
                                   prompt=prompt))
        diff_features = Inference(ft, img_size, ref_positions)
     
        for part, idx in part_idx.items():
            save_dir = os.path.join(target_dir, part, domain)
            os.makedirs(save_dir, exist_ok=True)
            part_diff = diff_features[idx[0]:idx[1]]
            part_clip = torch.mean(torch.cat(clip_features[part]), dim=0, keepdim=True)
            torch.save(part_diff, os.path.join(save_dir, "diff_feature.pt"))
            torch.save(part_clip, os.path.join(save_dir, "clip_feature.pt"))


