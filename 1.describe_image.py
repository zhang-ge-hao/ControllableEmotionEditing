from PIL import Image
import os
import time
import json
import base64
from tqdm import tqdm
from onevision import get_model, process

input_dir = "data/0.ilsvrc"
output_dir = "data/1.description"

os.makedirs(output_dir, exist_ok=True)

model, processor = get_model()

prompt = """Describe the picture briefly.

The description included two parts.

The first part is a description of the main components. Describe LESS THAN 4 most important components.

The second part is a description of the main actions. Describe LESS THAN 4 most important actions in the image.

Describe both components and actions using a numbered list. Each sentence contains LESS THAN 20 words.

Format:
```markdown
Components:

1. <COMPONENT_1>
1. <COMPONENT_2>
1. <COMPONENT_3>
1. <COMPONENT_4>

Actions:

1. <ACTION_1>
1. <ACTION_2>
1. <ACTION_3>
1. <ACTION_4>
```"""

for file_name in tqdm(list(os.listdir(input_dir))):
    file_path = os.path.join(input_dir, file_name)
    output_file_path = os.path.join(output_dir, f"{file_name}.txt")
    description = process(model, processor, file_path, prompt)
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(description)
