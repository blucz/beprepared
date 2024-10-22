# Captioning

## GPT4oCaption

Generates captions for images using GPT4o.

You must set `OPENAI_API_KEY` in your environment to use this node.

## Parameters

- `target_prop` (default: `'caption'`): The property to store the caption in
- `caption_type` (default: `'descriptive'`): The type of caption to generate (`'descriptive'` or `'booru'`)
- `prompt` (default: `None`): The prompt to use for the GPT-4o model (read the code if you're curious about customizing this)
- `instructions` (default: `None`): Additional instructions to append to the prompt
- `parallel` (default: `8`): The number of images to process in parallel. Adjust based on OpenAI rate limits.

### Output properties

- `image.{target_prop}`: The caption generated for the image

### Example

```python
dataset >> Gpt4oCaption(target_prop='caption', caption_type='descriptive')
```
