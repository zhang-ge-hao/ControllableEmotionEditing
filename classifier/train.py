from transformers import AutoModelForImageClassification, AutoFeatureExtractor, TrainingArguments, Trainer
from datasets import load_dataset
from accelerate import Accelerator
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from EmoSetBuilder import load_emo_set


dataset = load_dataset("zhang-ge-hao/EmoSet-118K-hf")
print(dataset)

model_name = "microsoft/resnet-18"
num_classes = 8

model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# dataset["train"] = dataset["train"].select(range(3200))
# dataset["validation"] = dataset["validation"].select(range(1600))

def preprocess_images(examples):
    images = [Image.fromarray(np.array(img)) for img in examples['img']]
    inputs = feature_extractor(images=images, return_tensors="pt")
    inputs['labels'] = examples['label']
    return inputs

train_dataset = dataset['train'].map(preprocess_images, batched=True)
validate_dataset = dataset['validation'].map(preprocess_images, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=70,
    per_device_eval_batch_size=70,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
)

# accelerator = Accelerator()

# model = accelerator.prepare(model)
# train_dataset = accelerator.prepare(train_dataset)
# validate_dataset = accelerator.prepare(validate_dataset)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

        posi_negi_labels = torch.div(labels, 4, rounding_mode='floor')
        posi_negi_probs = torch.stack([probs[:, :4], probs[:, 4:]])
        current_polarity_probs = posi_negi_probs[posi_negi_labels, torch.arange(len(labels))]
        another_polarity_probs = posi_negi_probs[1 - posi_negi_labels, torch.arange(len(labels))]

        true_probs = probs[torch.arange(len(labels)), labels].view(-1, 1)

        labels_local = labels % 4
        other_probs = current_polarity_probs + 0.1
        other_probs[torch.arange(len(labels_local)), labels_local] -= 0.1
        loss_1 = torch.mean(F.relu(other_probs - true_probs))

        other_probs = another_polarity_probs + 0.3
        loss_2 = torch.mean(F.relu(other_probs - true_probs))

        loss = loss_1 + loss_2

        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validate_dataset,
)

trainer.train()

trainer.save_model("./finetuned_resnet18")