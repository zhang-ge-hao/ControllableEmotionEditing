import os
from tqdm import tqdm
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, pipeline

from skimage import io, img_as_float, transform
from skimage.metrics import structural_similarity as ssim
import numpy as np
import json
from tqdm import tqdm

import json
import pandas as pd

def jsonl_to_excel(jsonl_file_path, excel_file_path):
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    
    df.to_excel(excel_file_path, index=False, engine='openpyxl')


classifier_path = "classifier/results/model_1/checkpoint-4726"

input_root_dir = "data/0.ilsvrc"
addition_root_dir = "data/1.addition"
output_dirs_and_titles = [
    ("data/4.edited_ip2p", "Ours"),
    ("data/5.ultra", "UltraEdit"),
    ("data/6.ip2p", "IP2P"),
    ("data/7.brush", "MagicBrush"),
]
emotion_label_list = [
    "amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"
]
class_count = 8

model_name = "classifier/results/model_1/checkpoint-4726"
base_model_name = "microsoft/resnet-18"
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_name)

classifier = pipeline("image-classification", model=model, feature_extractor=feature_extractor, device_map="auto", batch_size=1024)

results = []

for emotion_idx, emotion in tqdm(list(enumerate(emotion_label_list))):
    result = {"Emotion": emotion}
    for output_root_dir, title in output_dirs_and_titles:
        input_file_idxs = []
        output_file_idxs_list = []
        image_files = []
        for file_name in sorted(os.listdir(os.path.join(input_root_dir, emotion))):
            file_name = file_name.split(".")[0]
            # input_file_path = os.path.join(addition_root_dir, emotion, file_name, "modified.JPEG")
            input_file_path = os.path.join(input_root_dir, emotion, f"{file_name}.JPEG")
            input_file_idxs.append(len(image_files))
            image_files.append(input_file_path)
            output_file_idxs_list.append([])
            output_dir = os.path.join(output_root_dir, emotion, file_name)
            for output_file_name in sorted(os.listdir(output_dir)):
                output_file_path = os.path.join(output_dir, output_file_name)
                output_file_idxs_list[-1].append(len(image_files))
                image_files.append(output_file_path)
        predictions = classifier(image_files, top_k=class_count)
        scores = []
        is_matched = []
        for prediction in predictions:
            max_score, strongest_emotion = 0, None
            for d in prediction:
                if d["label"] == f"LABEL_{emotion_idx}":
                    scores.append(d["score"])
                if max_score < d["score"]:
                    max_score = d["score"]
                    strongest_emotion = d["label"]
            is_matched.append(strongest_emotion == f"LABEL_{emotion_idx}")

        delta_list = []
        for input_file_idx, output_file_idxs in zip(input_file_idxs, output_file_idxs_list):
            input_score = scores[input_file_idx]
            # if not is_matched[input_file_idx]:
            for output_file_idx in output_file_idxs:
                output_score = scores[output_file_idx]
                delta_list.append(output_score - input_score)
        
        ssim_list = []
        for file_name in sorted(os.listdir(os.path.join(input_root_dir, emotion))):
            file_name = file_name.split(".")[0]
            input_file_path = os.path.join(input_root_dir, emotion, f"{file_name}.JPEG")
            input_image = io.imread(input_file_path, as_gray=True)
            input_image = transform.resize(input_image, (512, 512), anti_aliasing=True)
            input_image_float = img_as_float(input_image)
            output_dir = os.path.join(output_root_dir, emotion, file_name)
            for output_file_name in sorted(os.listdir(output_dir)):
                output_file_path = os.path.join(output_dir, output_file_name)
                output_image = io.imread(output_file_path, as_gray=True)
                output_image = transform.resize(output_image, (512, 512), anti_aliasing=True)
                output_image_float = img_as_float(output_image)
                ssim_score = ssim(input_image_float, output_image_float, full=True, data_range=1.0)[0]
                ssim_list.append(ssim_score)
        success_delta_list = [d for d in delta_list if d > 0]
        result[f"Success Rate ({title})"] = sum(success_delta_list) / len(delta_list)
        result[f"Delta ({title})"] = sum(delta_list) / len(delta_list) if len(delta_list) > 0 else 0
        result[f"Success Delta ({title})"] = sum(success_delta_list) / len(success_delta_list) if len(success_delta_list) > 0 else 0
        result[f"SSIM ({title})"] = sum(ssim_list) / len(ssim_list) if len(ssim_list) > 0 else 0
        result[f"Delta_{{SSIM}} ({title})"] = sum([d * s for d, s in zip(delta_list, ssim_list)]) / len(delta_list) if len(delta_list) > 0 else 0
    results.append(result)

with open("data/results.jsonl", "w") as file:
    for result in results:
        file.write(json.dumps(result) + "\n")

jsonl_to_excel("data/results.jsonl", "data/results.xlsx")