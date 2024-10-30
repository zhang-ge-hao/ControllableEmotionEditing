from transformers import AutoModelForImageClassification, AutoFeatureExtractor, pipeline
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import os, json

base_model_name = "microsoft/resnet-18"
output_dir="./results"

dataset = load_dataset("zhang-ge-hao/EmoSet-118K-hf")
dataset["validation"] = dataset["validation"]
test_dataset = dataset['validation']

step_threshold = 100

checkpoint_names = os.listdir(output_dir)
checkpoint_names = [(n, int(n.split("-")[1])) for n in checkpoint_names]
checkpoint_names = [(n, i) for n, i in checkpoint_names if i > step_threshold]
checkpoint_names.sort(key=lambda p: -p[-1])
checkpoint_names = [n for n, _ in checkpoint_names]

for checkpoint_name in checkpoint_names:
    print(f"Checkpoint: {checkpoint_name}")

    checkpoint_dir = os.path.join(output_dir, checkpoint_name)
    model_name = checkpoint_dir

    model = AutoModelForImageClassification.from_pretrained(model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_name)
    classifier = pipeline("image-classification", model=model, feature_extractor=feature_extractor, device_map="auto")

    true_labels = []
    predicted_labels = []

    for example in tqdm(test_dataset):
        image = example['img']
        true_label = example['label']

        predictions = classifier(image)
        predicted_label = int(predictions[0]['label'].replace('LABEL_', ''))

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='micro')

    accuracy = accuracy_score(true_labels, predicted_labels)

    result_dict = {"accuracy": accuracy, 
                   "macro": {
                       "precision": precision_macro,
                       "recall": recall_macro,
                       "f1": f1_macro,
                   },
                   "micro": {
                       "precision": precision_micro,
                       "recall": recall_micro,
                       "f1": f1_micro,
                   }}

    with open(os.path.join(checkpoint_dir, "validation.json"), "w") as file:
        json.dump(result_dict, file)
    print(json.dumps(result_dict))