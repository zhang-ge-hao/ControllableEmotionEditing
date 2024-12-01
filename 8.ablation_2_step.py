from __future__ import annotations

import os
import k_diffusion as K
from omegaconf import OmegaConf
from tqdm import tqdm

from ip2p import Args, load_model_from_config, CFGDenoiser, edit


input_root_dir = "data/0.ilsvrc"
addition_root_dir = "data/1.addition"
output_root_dir = "data/8.2_step"

template_path = "templates/2_step_template.md"

count = 2

with open(template_path) as file:
    template = "".join(file.readlines())

emotion_label_list = [
    "amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"
]

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

        output_dir = os.path.join(output_root_dir, emotion, file_name)
        os.makedirs(output_dir, exist_ok=True)

        prompt = template.format(emotion=emotion)

        for output_idx in range(count):
            output_path = os.path.join(output_dir, f"{output_idx}.JPEG")
            if not os.path.exists(output_path):
                args.input, args.output, args.edit = file_path, output_path, prompt
                edit(model, null_token, model_wrap, model_wrap_cfg, args)