from PIL import Image
import os
import time
import json
import ollama
from ollama import Client
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
    required_keys_1 = {"amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"}
    if set(parsed_dict.keys()) == required_keys_1 \
        and all(isinstance(value, list) and len(value) == 3 for value in parsed_dict.values()):
        return parsed_dict, yaml_content
    else:
        return None, None

input_dir = "data/0.ilsvrc"
output_dir = "data/4.instruction"
inst_demo_dir = "data/2.addition"
generate_times = 5
retry_times = 20

os.makedirs(output_dir, exist_ok=True)

model, processor = get_model()

with open("templates/obtain_instruction.md") as file:
    prompt_template = "".join(file.readlines())

for file_idx, file_name in tqdm(list(enumerate(os.listdir(input_dir)))):
    file_path = os.path.join(input_dir, file_name)
    with open(os.path.join(inst_demo_dir, f"{file_name}.txt")) as file:
        inst_demo = "".join(file.readlines())
        inst_demo = yaml.safe_load(inst_demo)
    del inst_demo["retain"]
    inst_demo = [f"- {line}" for line in inst_demo.values()]
    inst_demo = "\n".join(inst_demo)

    for generate_idx in tqdm(list(range(generate_times))):
        for try_time in range(retry_times):
            output_file_path = os.path.join(output_dir, f"{file_name}.{generate_idx}.txt")
            
            response = process(model, processor, file_path, prompt_template.format(instruction_examples=inst_demo))
            
            _, instruction = parse_yaml_string(response)
            
            if instruction is None:
                if try_time == retry_times - 1:
                    raise RuntimeError(f"Retries exceeded for {file_name}")
                continue
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(instruction)
            break
