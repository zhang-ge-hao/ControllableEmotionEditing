import os
import shutil
import random
from tqdm import tqdm
import ollama
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, pipeline


output_path = "data/0.ilsvrc"
dataset_path = "/work/pi_juanzhai_umass_edu/gehaozhang/ILSVRC/Data/DET/test/"
select_count = 100

if os.path.exists(output_path):
    shutil.rmtree(output_path)

file_names = os.listdir(dataset_path)

random.shuffle(file_names)

emotion_label_list = [
    "amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"
]
class_count = 8
threshold = 0.5

model_name = "classifier/results/model_1/checkpoint-4726"
base_model_name = "microsoft/resnet-18"
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_name)

classifier = pipeline("image-classification", model=model, feature_extractor=feature_extractor, device_map="auto", batch_size=1024)

image_idx = 0
for emotion_idx, emotion in enumerate(emotion_label_list):
    os.makedirs(os.path.join(output_path, emotion), exist_ok=True)
    selected_count = 0
    while selected_count < select_count:
        file_name = file_names[image_idx]
        image_idx += 1
        file_path = os.path.join(dataset_path, file_name)
        predictions = classifier([file_path], top_k=class_count)
        prediction = predictions[0]
        score = [prediction[i]["score"] for i in range(len(prediction)) if f"LABEL_{emotion_idx}" == prediction[i]["label"]][0]
        if score < threshold:
            shutil.copy(
                os.path.join(dataset_path, file_name),
                os.path.join(output_path, emotion, file_name)
            )
            selected_count += 1