import os
from tqdm import tqdm
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, pipeline

from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
import numpy as np

classifier_path = "classifier/results/model_1/checkpoint-4726"

input_dir = "data/0.ilsvrc"
naive_dir = "data/8.naiveE"
output_dir = "data/5.emotionE"

count = 15
class_count = 8

model_name = "classifier/results/model_1/checkpoint-4726"
base_model_name = "microsoft/resnet-18"
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_name)

classifier = pipeline("image-classification", model=model, feature_extractor=feature_extractor, device_map="auto", batch_size=2 * count + 1)

for file_name in list(os.listdir(input_dir)):
    folder_name = file_name.split(".")[0]
    
    input_image = os.path.join(input_dir, file_name)
    output_images = [os.path.join(output_dir, folder_name, "amusement", f"{file_idx}.JPEG") for file_idx in range(count)]
    naive_images = [os.path.join(naive_dir, folder_name, "amusement", f"{file_idx}.JPEG") for file_idx in range(count)]

    predictions = classifier([input_image] + output_images + naive_images, top_k=class_count)

    input_amusement_score = [d["score"] for d in predictions[0] if d["label"] == "LABEL_0"][0]
    output_amusement_scores = [[d["score"] for d in prediction if d["label"] == "LABEL_0"][0] for prediction in predictions[1: 1 + count]]
    naive_amusement_scores = [[d["score"] for d in prediction if d["label"] == "LABEL_0"][0] for prediction in predictions[1 + count: ]]

    d_output_amusement_scores = [score - input_amusement_score for score in output_amusement_scores]
    d_naive_amusement_scores = [score - input_amusement_score for score in naive_amusement_scores]

    input_image_float = img_as_float(io.imread(input_image, as_gray=True))
    output_images_float_list = [img_as_float(io.imread(output_image, as_gray=True)) for output_image in output_images]
    naive_images_float_list = [img_as_float(io.imread(naive_image, as_gray=True)) for naive_image in naive_images]

    output_ssim_scores = [ssim(input_image_float, i, full=True, data_range=1.0)[0] for i in output_images_float_list]
    naive_ssim_scores = [ssim(input_image_float, i, full=True, data_range=1.0)[0] for i in naive_images_float_list]

    print("delta score output: ", sum(d_output_amusement_scores) / count)
    print("delta score naive:  ", sum(d_naive_amusement_scores) / count)

    print("ssim output: ", sum(output_ssim_scores) / count)
    print("ssim naive:  ", sum(naive_ssim_scores) / count)

    print("product output: ", sum(x * y for x, y in zip(d_output_amusement_scores, output_ssim_scores)) / count)
    print("product naive:  ", sum(x * y for x, y in zip(d_naive_amusement_scores, naive_ssim_scores)) / count)

    print("=" * 20)