from __future__ import annotations

from PIL import Image

import os
import k_diffusion as K
from omegaconf import OmegaConf
from tqdm import tqdm
import yaml
import time

# from ultra_edit import get_pipeline, edit
from ip2p import Args, load_model_from_config, CFGDenoiser, edit

def mask_reverse(mask_path):
    img = Image.open(mask_path).convert("L")
    inverted_img = Image.eval(img, lambda x: 255 - x)
    inverted_img = inverted_img.convert("RGB")
    return inverted_img

input_root_dir = "data/0.ilsvrc"
addition_root_dir = "data/1.addition"
instruction_root_dir = "data/3.instruct"
output_root_dir = "data/4.edited_ip2p"

emotion_label_list = [
    "amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"
]

# pipeline = get_pipeline()
args = Args()
config = OmegaConf.load(args.config)
model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
model.eval().cuda()
model_wrap = K.external.CompVisDenoiser(model)
model_wrap_cfg = CFGDenoiser(model_wrap)
null_token = model.get_learned_conditioning([""])


for emotion in emotion_label_list:
    for file_name in tqdm(os.listdir(os.path.join(input_root_dir, emotion))):
        file_name = file_name.split(".")[0]
        file_path = os.path.join(addition_root_dir, emotion, file_name, "modified.JPEG")
        mask_path = os.path.join(addition_root_dir, emotion, file_name, "mask.JPEG")
        mask_image = mask_reverse(mask_path)

        output_dir = os.path.join(output_root_dir, emotion, file_name)
        instruction_dir = os.path.join(instruction_root_dir, emotion, file_name)
        os.makedirs(output_dir, exist_ok=True)
        for instruction_file_name in os.listdir(instruction_dir):
            instruction_path = os.path.join(instruction_dir, instruction_file_name)
            output_image_path = os.path.join(output_dir, f"{instruction_file_name}.JPEG")
            with open(instruction_path) as file:
                instruction = "".join(file.readlines())
            instruction = f"Bring {emotion} in this image: {instruction}"
            if not os.path.exists(output_image_path):
                # edit(pipeline, file_path, output_image_path, instruction, mask_image)
                args.input, args.output, args.edit = file_path, output_image_path, instruction
                edit(model, null_token, model_wrap, model_wrap_cfg, args)
            else:
                print("Cached: " + output_image_path)