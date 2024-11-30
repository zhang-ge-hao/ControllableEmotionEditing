from PIL import Image, ImageDraw, ImageFilter
import os
import time
import json
import ollama
import base64
from tqdm import tqdm
import time
import cv2

import yaml
import re
import sys

LISA_path = "../LISA"

if LISA_path not in sys.path:
    sys.path.insert(0, LISA_path)

from api import init, inference, parse_args

import random
import numpy as np
import shutil

from onevision import get_model, process
from ultra_edit import get_pipeline, edit


def calculate_white_ratio(image_path):
    image = Image.open(image_path).convert('L')
    binary = np.array(image) > 128
    ones = np.ones_like(image)

    return np.sum(binary) / np.sum(ones)

def apply_mask(image_path, mask_path, output_path):
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_width, image_height = image.size
    blur_mask = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(blur_mask)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, image_width), min(y + h, image_height)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=15))
    result = Image.composite(blurred_image, image, blur_mask)
    result = result.convert("RGB")
    result.save(output_path)

random.seed(199907)

input_root_dir = "data/0.ilsvrc"
output_root_dir = "data/1.addition"

emotion_label_list = [
    "amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"
]
direction_short_list = [
    "add", "remove", "modify", "retain"
]

direction_long_list = [
    "to add a random element into this image",
    "to remove a random element into this image",
    "to modify a random element to another random element",
    "to retain a random component in the image",
]

format_list = [
    "Add <SOMETHING> in <LOCATION>.",
    "Remove <SOMETHING>.",
    "Modify <SOMETHING> to <ANOTHER_THING>.",
    "Retain <SOMETHING>.",
]

direction_tuples = list(zip(direction_short_list, direction_long_list, format_list))

model, processor = get_model()

mask_args, clip_image_processor, mask_transform, mask_tokenizer, mask_model = init(parse_args([]))

pipeline = get_pipeline()

with open("templates/random_addition.md") as file:
    random_addition_template = "".join(file.readlines())
with open("templates/generate_mask.md") as file:
    generate_mask_template = "".join(file.readlines())

for emotion in emotion_label_list:
    for file_name in tqdm(os.listdir(os.path.join(input_root_dir, emotion))):
        file_name = file_name.split(".")[0]
        file_path = os.path.join(input_root_dir, emotion, f"{file_name}.JPEG")

        random.shuffle(direction_tuples)
        direction_short, direction_long, format = direction_tuples[0]

        output_dir = os.path.join(output_root_dir, emotion, file_name)
        os.makedirs(output_dir, exist_ok=True)
        instruction_file_name = "instruction"
        component_file_name = f"component"
        mask_file_name = "mask.JPEG"
        masked_file_name = "masked.JPEG"
        modified_file_name = "modified.JPEG"

        with open(os.path.join(output_dir, "direction"), "w") as file:
            file.write(direction_short)

        while True:
            # addition instruction generation
            prompt = random_addition_template.format(direction_long=direction_long, format=format)
            response = process(model, processor, file_path, prompt)
            with open(os.path.join(output_dir, instruction_file_name), "w", encoding="utf-8") as f:
                f.write(response)

            # mask generation
            inference(generate_mask_template.format(instruction=response), 
                    file_path, os.path.join(output_dir, mask_file_name), 
                    mask_args, clip_image_processor, mask_transform, mask_tokenizer, mask_model)
            
            if calculate_white_ratio(os.path.join(output_dir, mask_file_name)) > 0.5:
                continue

            # blur mask
            apply_mask(file_path, os.path.join(output_dir, mask_file_name), 
                    os.path.join(output_dir, masked_file_name))

            # modify
            modified_file_path = os.path.join(output_dir, modified_file_name)
            if direction_short != "retain":
                edit(pipeline, file_path, modified_file_path, response)
            else:
                shutil.copy(file_path, modified_file_path)

            break
