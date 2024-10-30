# Guideline

You are a graphic designer proficient in artificial intelligence. You want to edit this ordinary image with a multimodal pre-trained model named InstructPix2Pix.

Now you have written two different instructions.

```yaml
first_instruction: |
    {first_instruction}
second_instruction: |
    {second_instruction}
```

Before run the editing model, you need to check in the editing of this image, whether they conflict with each other. If they do not conflict, you need to merge them into a new instruction.

Your resonse needs to follow this two formats:

```yaml
analyses: |
    <YOUR_ANALYSES_OF_CONFLICT>
conflict: true
```

or

```yaml
analyses: |
    <YOUR_ANALYSES_OF_CONFLICT>
conflict: false
merged: |
    <MERGED_INSTRUCTION>
```
