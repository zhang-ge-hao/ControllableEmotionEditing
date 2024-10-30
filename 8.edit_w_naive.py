from __future__ import annotations

import os
import k_diffusion as K
from omegaconf import OmegaConf
from tqdm import tqdm

from ultra_edit import get_pipeline, edit
# from ip2p import Args, load_model_from_config, CFGDenoiser, edit


def main():
    input_dir = "data/0.ilsvrc"
    output_dir = "data/8.naiveE"

    instructions = [
        "Give this image more amusement.",
        "Make it amusement.",
        "Edit the image to amuse the viewers.",
        "Be amused.",
        "Let it amusement"
    ]

    count = 15

    # args = Args()

    # config = OmegaConf.load(args.config)
    # model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    # model.eval().cuda()
    # model_wrap = K.external.CompVisDenoiser(model)
    # model_wrap_cfg = CFGDenoiser(model_wrap)
    # null_token = model.get_learned_conditioning([""])
    pipeline = get_pipeline()

    for file_name in tqdm(list(os.listdir(input_dir))):
        file_path = os.path.join(input_dir, file_name)
        for output_idx in range(count):
            instruction = instructions[output_idx % len(instructions)]

            os.makedirs(os.path.join(output_dir, file_name.split(".")[0], "amusement"), exist_ok=True)

            output_path = os.path.join(output_dir, file_name.split(".")[0], "amusement", f"{output_idx}.JPEG")

            # args.input, args.output, args.edit = file_path, output_path, instruction
            # edit(model, null_token, model_wrap, model_wrap_cfg, args)
            if not os.path.exists(output_path):
                edit(pipeline, file_path, output_path, instruction)
            else:
                print("Cached: " + output_path)


if __name__ == "__main__":
    main()
