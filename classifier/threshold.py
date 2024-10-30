import json
import matplotlib.pyplot as plt

test_result_path = "results/checkpoint-1014/test_batch.jsonl"
output_fig_path = "results/checkpoint-1014/threshold.png"
granularity = 20
label_count = 8

with open(test_result_path) as file:
    test_result = [json.loads(line.strip()) for line in file]

label2line = {label: {"threshold": [], "recall": [], "precesion": []} for label in range(8)}

for threshold_raw in range(granularity):
    threshold = threshold_raw / granularity
    for label in range(label_count):
        true_positive_count = len([d for d in test_result if d["true_label"] == label])
        predict_positive_count = len([d for d in test_result if d["predicted_label"] == label and d["score"] > threshold])
        tp = len([d for d in test_result if d["true_label"] == label and d["predicted_label"] == label and d["score"] > threshold])
        recall = tp / true_positive_count
        precesion = (tp / predict_positive_count) if predict_positive_count > 0 else 0
        label2line[label]["threshold"].append(threshold)
        label2line[label]["recall"].append(recall)
        label2line[label]["precesion"].append(precesion)

# print(json.dumps(label2line, indent=4))

fig, axes = plt.subplots(4, 2, figsize=(12, 12))
fig.suptitle("Recall and Precision per Label at Different Thresholds")

for label, ax in enumerate(axes.flat):
    data = label2line[label]
    ax.plot(data["threshold"], data["recall"], label="Recall", marker='o')
    ax.plot(data["threshold"], data["precesion"], label="Precision", marker='s')
    ax.set_ylim(0.2, 1.0)  # Fix y-axis range
    ax.axhline(0.9, color='gray', linestyle='--', linewidth=1)  # Add dashed line at y=0.9
    ax.set_title(f"Label {label}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Percentage")
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_fig_path)