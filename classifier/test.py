from transformers import AutoModelForImageClassification, AutoFeatureExtractor, pipeline
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import os, json

model_name = "results/checkpoint-1014"
base_model_name = "microsoft/resnet-18"

batch_size = 500

model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_name)

classifier = pipeline("image-classification", model=model, feature_extractor=feature_extractor, device_map="auto", batch_size=batch_size)

dataset = load_dataset("zhang-ge-hao/EmoSet-118K-hf")
test_dataset = dataset['test']

true_labels = []
predicted_labels = []
scores = []

with open(os.path.join(model_name, "test_batch.jsonl"), "w") as file:
    batch_images = []
    batch_true_labels = []

    for example in tqdm(test_dataset):
        image = example['img']
        true_label = example['label']
        batch_images.append(image)
        batch_true_labels.append(true_label)

        if len(batch_images) == batch_size:
            predictions = classifier(batch_images)

            for idx, prediction in enumerate(predictions):
                predicted_label = int(prediction[0]['label'].replace('LABEL_', ''))
                score = prediction[0]['score']

                true_labels.append(batch_true_labels[idx])
                predicted_labels.append(predicted_label)
                scores.append(score)

                file.write(json.dumps({
                    "true_label": batch_true_labels[idx],
                    "predicted_label": predicted_label,
                    "score": score
                }) + "\n")

            batch_images = []
            batch_true_labels = []

    if batch_images:
        predictions = classifier(batch_images)
        for idx, prediction in enumerate(predictions):
            predicted_label = int(prediction[0]['label'].replace('LABEL_', ''))
            score = prediction[0]['score']

            true_labels.append(batch_true_labels[idx])
            predicted_labels.append(predicted_label)
            scores.append(score)

            file.write(json.dumps({
                "true_label": batch_true_labels[idx],
                "predicted_label": predicted_label,
                "score": score
            }) + "\n")

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='micro')
accuracy = accuracy_score(true_labels, predicted_labels)

print("Accuracy:")
print(accuracy)

print("\nMacro Average Metrics:")
print(f"Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1-Score: {f1_macro:.4f}")

print("\nMicro Average Metrics:")
print(f"Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1-Score: {f1_micro:.4f}")