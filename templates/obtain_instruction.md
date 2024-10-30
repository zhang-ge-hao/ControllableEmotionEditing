# Guideline

You are a graphic designer proficient in artificial intelligence. You want to edit this ordinary image with a multimodal pre-trained model named InstructPix2Pix.

Here are some examples of instructions:

{instruction_examples}

Your goal is to edit images to inspire different emotions in the viewer without big modifications to the layout.

Target emotions:

1. amusement - the state or experience of finding something funny.
2. awe - a feeling of reverential respect mixed with fear or wonder.
3. contentment - a state of happiness and satisfaction.
4. excitement - a feeling of great enthusiasm and eagerness.
5. anger - a strong feeling of annoyance, displeasure, or hostility.
6. disgust - a feeling of revulsion or strong disapproval aroused by something unpleasant or offensive.
7. fear - an unpleasant emotion caused by the belief that someone or something is dangerous, likely to cause pain, or a threat.
8. sadness - the condition or quality of being sad.

Write down your instructions to be used in InstructPix2Pix about editing this image.

# Note

1. Generate 3 instructions for each emotion.
2. Each instruction you return needs to be succinct. Contains only a single sentence with NO MORE THAN 15 words.
3. Clarify your needs, for the model can only understand simple instructions.
4. About the face of a human or animal, you can ONLY do SLIGHT modifications!

# Format

The returned result needs to be in YAML format. The format is below. Do not provide ANY additional content, e.g., further explanation, markdown titles, etc.

```yaml
amusement:
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
awe:
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
contentment:
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
excitement:
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
anger:
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
disgust:
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
fear:
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
sadness:
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
    - "<YOUR_INSTRUCTION>"
```