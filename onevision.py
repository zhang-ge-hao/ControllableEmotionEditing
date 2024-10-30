# Import necessary libraries
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


def get_model():
    # Load the model
    model_id = "llava-hf/llava-onevision-qwen2-7b-si-hf"
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)

    # Load the processor
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor


def process(model, processor, image_path, prompt):
    # Define chat history and format prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id

    # Load image
    raw_image = Image.open(image_path)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    # Generate output
    output = model.generate(**inputs, max_new_tokens=2048, do_sample=True, pad_token_id=pad_token_id)

    # Decode and print only the generated text result
    generated_text = processor.decode(output[0], skip_special_tokens=True).split("assistant\n")[1: ]
    generated_text = "assistant\n".join(generated_text)
    
    return generated_text
