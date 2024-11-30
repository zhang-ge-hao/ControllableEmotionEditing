from PIL import Image
import os
import time
import json
import ollama
from ollama import Client
import base64
from tqdm import tqdm
import time

import random

import yaml
import re

from onevision import get_model, process

random.seed(199907)

input_root_dir = "data/0.ilsvrc"
addition_root_dir = "data/1.addition"
example_root_dir = "data/2.example"
output_root_dir = "data/3.instruct"

generate_times = 2

demo_instruction_types = ["add", "remove", "modify", "color"]

emotion_label_list = [
    "amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"
]

emotion_desc_list = [
    "amusement - the state or experience of finding something funny.",
    "awe - a feeling of reverential respect mixed with fear or wonder.",
    "contentment - a state of happiness and satisfaction.",
    "excitement - a feeling of great enthusiasm and eagerness.",
    "anger - a strong feeling of annoyance, displeasure, or hostility.",
    "disgust - a feeling of revulsion or strong disapproval aroused by something unpleasant or offensive.",
    "fear - an unpleasant emotion caused by the belief that someone or something is dangerous, likely to cause pain, or a threat.",
    "sadness - the condition or quality of being sad.",
]

model, processor = get_model()

with open("templates/obtain_instruct.md") as file:
    prompt_template = "".join(file.readlines())

for emotion_label, emotion_desc in zip(emotion_label_list, emotion_desc_list):
    for file_name in tqdm(os.listdir(os.path.join(input_root_dir, emotion_label))):
        file_name = file_name.split(".")[0]
        file_path = os.path.join(addition_root_dir, emotion_label, file_name, "modified.JPEG")

        # get example
        examples = []
        for demo_instruction_type in demo_instruction_types:
            example_dir = os.path.join(example_root_dir, emotion_label, file_name, demo_instruction_type)
            for example_file_name in os.listdir(example_dir):
                with open(os.path.join(example_dir, example_file_name)) as file:
                    example = "".join(file.readlines()).strip()
                if demo_instruction_type == "color":
                    example = f"Give this image a {example} tint."
                examples.append(example)

        output_dir = os.path.join(output_root_dir, emotion_label, file_name)
        os.makedirs(output_dir, exist_ok=True)
        responses = []
        for generate_idx in range(generate_times):
            random.shuffle(examples)
            prompt = prompt_template.format(emotion_label=emotion_label, 
                                            emotion_desc=emotion_desc, 
                                            examples="\n".join(examples))
            response = process(model, processor, file_path, prompt)
            with open(os.path.join(output_dir, str(generate_idx)), "w") as file:
                file.write(response)
