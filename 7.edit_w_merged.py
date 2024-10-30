from __future__ import annotations

import os
import k_diffusion as K
from omegaconf import OmegaConf
from tqdm import tqdm
import yaml

from ultra_edit import get_pipeline, edit
# from ip2p import Args, load_model_from_config, CFGDenoiser, edit

def main():
    input_dir = "data/0.ilsvrc"
    instruction_dir = "data/6.filtered"
    output_dir = "data/7.filteredE"

    # args = Args()
    # config = OmegaConf.load(args.config)
    # model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    # model.eval().cuda()
    # model_wrap = K.external.CompVisDenoiser(model)
    # model_wrap_cfg = CFGDenoiser(model_wrap)
    # null_token = model.get_learned_conditioning([""])
    pipeline = get_pipeline()

    for file_name in tqdm(list(os.listdir(input_dir))):
        original_image_path = os.path.join(input_dir, file_name)
        emotion_idx_map = {}
        output_folder_name = file_name.split(".")[0]

        instruction_path = os.path.join(instruction_dir, f"{file_name}.txt")

        with open(instruction_path) as file:
            instruction_dict = yaml.safe_load("".join(file.readlines()))

        for emotion, instructions in instruction_dict.items():

            if emotion != "amusement":
                continue

            assert all(isinstance(instruction, str) for instruction in instructions)
            if emotion not in emotion_idx_map:
                emotion_idx_map[emotion] = 0
            for instruction in instructions:
                os.makedirs(os.path.join(output_dir, output_folder_name, emotion), exist_ok=True)

                output_image_path = os.path.join(output_dir, output_folder_name, emotion, f"{emotion_idx_map[emotion]}.JPEG")
                emotion_idx_map[emotion] += 1

                # args.input, args.output, args.edit = original_image_path, output_image_path, instruction
                # edit(model, null_token, model_wrap, model_wrap_cfg, args)
                edit(pipeline, original_image_path, output_image_path, instruction)


if __name__ == "__main__":
    main()
