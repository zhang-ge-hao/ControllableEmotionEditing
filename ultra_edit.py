# For Editing with SD3
import os
import torch
from diffusers import StableDiffusion3InstructPix2PixPipeline
from diffusers.utils import load_image
import requests
import PIL.Image


def get_pipeline():
    pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained("BleachNick/SD3_UltraEdit_w_mask", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def edit(pipe, input, output, prompt, mask: PIL.Image=None, image_guidance_scale=1.5, guidance_scale=7.5):
    img = load_image(input)
    ori_size = img.size
    img = img.resize((512, 512))
    # mask_img = load_image("mask_img.png").resize(img.size)
    # For free form Editing, seed a blank mask
    if mask is None:
        mask_img = PIL.Image.new("RGB", img.size, (255, 255, 255))
    else:
        mask_img = mask.resize(img.size)
    image = pipe(
        prompt,
        image=img,
        mask_img=mask_img,
        negative_prompt="",
        num_inference_steps=50,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
    ).images[0]
    image = image.resize(ori_size)
    image.save(output)
    # display image