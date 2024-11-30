from PIL import Image
import os
import time
import json
import ollama
from ollama import Client
import base64
from tqdm import tqdm
import time

import yaml
import numpy as np
import sys

LISA_path = "../LISA"

if LISA_path not in sys.path:
    sys.path.insert(0, LISA_path)

from api import init, inference, parse_args

from onevision import get_model, process

input_root_dir = "data/0.ilsvrc"
addition_root_dir = "data/1.addition"
output_root_dir = "data/2.example"

open_direction_generate_times = 2
close_direction_generate_times = 1

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

colors = ["Red", "Blue", "Green", "Yellow", "Gray", "Pink", "Purple", "Orange"]

open_directions = ["add", "remove", "modify"]
close_directions = ["color", "brightness"]

direction_tuples = [
    (
        "add", # label
        "Add <SOMETHING> in <LOCATION>.", # format
        "add something", # action
        "What you think should be added", # question
    ), 
    (
        "remove",
        "Remove <SOMETHING>.",
        "remove something",
        "What you think should be removed",
    ), 
    (
        "modify", 
        "Modify <SOMETHING> to <ANOTHER_THING>.",
        "modify something",
        "What you think should be modified",
    ), 
    (
        "color", 
        f"Your response would ONLY contain ONE word chosen from these: {', '.join(colors)}.",
        "adjusting the color tint",
        "What color tint you think suitable",
    )
]

model, processor = get_model()

mask_args, clip_image_processor, mask_transform, mask_tokenizer, mask_model = init(parse_args([]))

with open("templates/obtain_example.md") as file:
    prompt_template = "".join(file.readlines())
with open("templates/generate_mask.md") as file:
    generate_mask_template = "".join(file.readlines())

for emotion_label, emotion_desc in zip(emotion_label_list, emotion_desc_list):
    for file_name in tqdm(os.listdir(os.path.join(input_root_dir, emotion_label))):
        file_name = file_name.split(".")[0]
        file_path = os.path.join(addition_root_dir, emotion_label, file_name, "masked.JPEG")

        addition_path = os.path.join(addition_root_dir, emotion_label, file_name, "instruction")
        addition_mask_path = os.path.join(addition_root_dir, emotion_label, file_name, "mask.JPEG")
        with open(addition_path) as file:
            addition = "".join(file.readlines())
        for direction, format, action, question in direction_tuples:
            output_dir = os.path.join(output_root_dir, emotion_label, file_name, direction)
            os.makedirs(output_dir, exist_ok=True)
            generate_times = open_direction_generate_times if direction in open_directions \
                else close_direction_generate_times
            generate_idx = 0
            while generate_idx < generate_times:
                prompt = prompt_template.format(emotion_label=emotion_label, 
                                                emotion_desc=emotion_desc, 
                                                format=format,
                                                action=action, 
                                                question=question)
                response = process(model, processor, file_path, prompt)
                if direction in open_directions:
                    matched = True
                    for ban_word in ["blur", "rectangle"]:
                        if ban_word in response.lower():
                            matched = False
                else:
                    assert direction == "color"
                    matched = False
                    for color in colors:
                        if color.lower() in response.lower():
                            response = color
                            matched = True
                with open(os.path.join(output_dir, f"{generate_idx}_instruct"), "w") as file:
                    file.write(response)
                if matched:
                    generate_idx += 1