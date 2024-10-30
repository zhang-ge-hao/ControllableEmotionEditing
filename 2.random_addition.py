from PIL import Image
import os
import time
import json
import ollama
import base64
from tqdm import tqdm
import time

import yaml
import re

from onevision import get_model, process

def parse_yaml_string(input_string):
    if "```yaml" in input_string:
        code_blocks = re.findall(r'```yaml(.*?)```', input_string, re.DOTALL)
        yaml_content = "\n".join(code_blocks).strip()
    elif "```YAML" in input_string:
        code_blocks = re.findall(r'```YAML(.*?)```', input_string, re.DOTALL)
        yaml_content = "\n".join(code_blocks).strip()
    elif "```Yaml" in input_string:
        code_blocks = re.findall(r'```Yaml(.*?)```', input_string, re.DOTALL)
        yaml_content = "\n".join(code_blocks).strip()
    elif "```" in input_string:
        code_blocks = re.findall(r'```(.*?)```', input_string, re.DOTALL)
        yaml_content = "\n".join(code_blocks).strip()
    else:
        yaml_content = input_string.strip()
    try:
        parsed_dict = yaml.safe_load(yaml_content)
    except yaml.YAMLError:
        return None, None
    if not isinstance(parsed_dict, dict):
        return None, None
    required_keys = {"add", "change", "retain", "delete"}
    if set(parsed_dict.keys()) == required_keys and all(isinstance(value, str) for value in parsed_dict.values()):
        return parsed_dict, yaml_content
    else:
        return None, None

input_dir = "data/0.ilsvrc"
output_dir = "data/2.addition"
retry_times = 20

os.makedirs(output_dir, exist_ok=True)

model, processor = get_model()

prompt = """You are a graphic designer proficient in artificial intelligence. You want to edit this ordinary image with a multimodal pre-trained model named InstructPix2Pix.

Write down your instructions to be used in InstructPix2Pix about editing this image.

There are four types of editing instructions for images:
- add
- change
- retain
- delete

Note: 
1. Provide one instruction for each of the four types.
2. Each instruction you return needs to be succinct. Contains only a single sentence with NO MORE THAN 15 words.
3. Clarify your needs, for the model can only understand simple instructions.
4. The returned result needs to be in YAML format. The format is below. Do not provide ANY additional content, e.g., further explanation, markdown titles, etc.
5. Change, retain, and delete operations NEED to be DIRECTLY related to existing components in the image.

```
add: "Add shiny ribbons in the background."
change: "Change the surface of the balloon to a fur texture."
retain: "Retain the stars in the sky."
delete: "Delete the flowers from the lawn."
```"""

for file_idx, file_name in tqdm(list(enumerate(os.listdir(input_dir)))):
    file_path = os.path.join(input_dir, file_name)
    output_file_path = os.path.join(output_dir, f"{file_name}.txt")
    selected_instruction_path = os.path.join(output_dir, f"{file_name}.s.txt")
    messages = [{"role": "user", "content": prompt, "images": [file_path]}]
    for try_time in range(retry_times):
        
        response = process(model, processor, file_path, prompt)
        
        instruction_dict, instruction = parse_yaml_string(response)

        if instruction is None:
            if try_time == retry_times - 1:
                raise RuntimeError(f"Retries exceeded for {file_name}")
            continue
        selected_instruction = instruction_dict[["add", "change", "retain", "delete"][file_idx % 4]]
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(instruction)
        with open(selected_instruction_path, "w") as f:
            f.write(selected_instruction)
        break
