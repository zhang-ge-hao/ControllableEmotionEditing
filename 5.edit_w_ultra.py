from __future__ import annotations

import os
import k_diffusion as K
from omegaconf import OmegaConf
from tqdm import tqdm

from ultra_edit import get_pipeline, edit


input_root_dir = "data/0.ilsvrc"
addition_root_dir = "data/1.addition"
output_root_dir = "data/5.ultra_direct"

template_path = "templates/baseline_template.md"

count = 2

with open(template_path) as file:
    template = "".join(file.readlines())

emotion_label_list = [
    "amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"
]
pipeline = get_pipeline()

for emotion in emotion_label_list:
    for file_name in tqdm(os.listdir(os.path.join(input_root_dir, emotion))):
        file_name = file_name.split(".")[0]
        file_path = os.path.join(input_root_dir, emotion, f"{file_name}.JPEG")

        addition_path = os.path.join(addition_root_dir, emotion, file_name, "instruction")
        with open(addition_path) as file:
            addition = "".join(file.readlines())

        output_dir = os.path.join(output_root_dir, emotion, file_name)
        os.makedirs(output_dir, exist_ok=True)

        prompt = template.format(emotion=emotion, addition=addition)

        for output_idx in range(count):
            output_path = os.path.join(output_dir, f"{output_idx}.JPEG")
            if not os.path.exists(output_path):
                edit(pipeline, file_path, output_path, prompt)
