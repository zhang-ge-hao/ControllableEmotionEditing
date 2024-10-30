import os
from tqdm import tqdm
import yaml
import copy
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

    if "analyses" not in parsed_dict or "conflict" not in parsed_dict:
        return None, None
    if not isinstance(parsed_dict["analyses"], str) or not isinstance(parsed_dict["conflict"], bool):
        return None, None
    if parsed_dict["conflict"]:
        return parsed_dict, yaml_content
    if "merged" not in parsed_dict or not isinstance(parsed_dict["merged"], str):
        return None, None
    return parsed_dict, yaml_content


def merge_instruction(model, processor, image_path, instruction, additional_instruction, retry_times=20):
    with open("templates/merge_instruction.md") as file:
        prompt_template = "".join(file.readlines())
    prompt = prompt_template.format(first_instruction=instruction, second_instruction=additional_instruction)

    for try_time in range(retry_times):
        result = process(model, processor, image_path, prompt)

        result_dict, result_str = parse_yaml_string(result)

        if result_dict is None:
            if try_time == retry_times - 1:
                raise RuntimeError(f"Retries exceeded for {file_name}")
            continue
        return result_dict


input_dir = "data/0.ilsvrc"
emotion_instruction_dir = "data/4.instruction"
additional_instruction_dir = "data/2.addition"
filtered_instruction_dir = "data/6.filtered"
generate_times = 5

os.makedirs(filtered_instruction_dir, exist_ok=True)

model, processor = get_model()

for file_idx, file_name in tqdm(list(enumerate(os.listdir(input_dir)))):
    file_path = os.path.join(input_dir, file_name)
    with open(os.path.join(additional_instruction_dir, f"{file_name}.s.txt")) as file:
        additional_instruction = "".join(file.readlines())
    emotion_instructions = None
    for generate_idx in range(generate_times):
        with open(os.path.join(emotion_instruction_dir, f"{file_name}.{generate_idx}.txt")) as file:
            instructions = yaml.safe_load("".join(file.readlines()))
        if generate_idx == 0:
            emotion_instructions = copy.deepcopy(instructions)
        else:
            assert set(instructions.keys()) == set(emotion_instructions.keys())
            for k, v in instructions.items():
                emotion_instructions[k].extend(v)
    filtered_emotion_instructions = {k: [] for k in emotion_instructions.keys()}
    log_file_path = os.path.join(filtered_instruction_dir, f"{file_name}.log.txt")
    with open(log_file_path, "w") as log_file:
        for k, v in tqdm(list(emotion_instructions.items())):
            for instruction in tqdm(v):
                merged_result = merge_instruction(model, processor, file_path, instruction, additional_instruction)
                
                merged_result["instruction"] = instruction
                merged_result["additional_instruction"] = additional_instruction
                log_file.write(yaml.dump(merged_result, default_flow_style=False) + "\n")

                if merged_result["conflict"]:
                    continue
                filtered_emotion_instructions[k].append(merged_result["merged"])
        output_path = os.path.join(filtered_instruction_dir, f"{file_name}.txt")
        with open(output_path, "w") as file:
            file.write(yaml.dump(filtered_emotion_instructions, default_flow_style=False))
