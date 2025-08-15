# Captioning

## GeminiCaption

Generates captions for images using Gemini 2.0 Flash Vision.

You must set either `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your environment to use this node.

## Parameters

- `target_prop` (default: `'caption'`): The property to store the caption in
- `prompt` (default: `None`): The prompt to use for the Gemini model
- `instructions` (default: `None`): Additional instructions to append to the prompt
- `parallel` (default: `8`): The number of images to process in parallel. Adjust based on API rate limits.

### Output properties

- `image.{target_prop}`: The caption generated for the image

### Example

```python
dataset >> GeminiCaption()
```

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
dataset >> Gpt4oCaption(caption_type='descriptive')
```

## JoyCaptionAlphaOne

Generates captions for images using JoyCaption Alpha One

### Parameters

- `target_prop` (default: `'caption'`): The property to store the caption in
- `caption_type` (default: `'descriptive'`): The type of caption to generate (`'descriptive'`, `'stable_diffusion'`, or `'booru'`)
- `caption_tone` (default: `'formal'`): The tone of the caption (`'formal'` or `'informal'`)
- `caption_length` (default: `'any'`): The length of the caption (`'any'` or an integer)
- `batch_size` (default: `4`): The number of images to process in parallel

### Output properties

- `image.{target_prop}`: The caption generated for the image

### Example

```python
dataset >> JoyCaptionAlphaOne
```

## JoyCaptionAlphaTwo

Generates captions for images using JoyCaption Alpha Two with additional caption types and options.

### Parameters

- `target_prop` (default: `'caption'`): The property to store the caption in
- `caption_type` (default: `'descriptive'`): The type of caption to generate. Options:
  - `'descriptive'`
  - `'descriptive_informal'`
  - `'training_prompt'`
  - `'midjourney'`
  - `'booru_tag_list'`
  - `'booru_like_tag_list'`
  - `'art_critic'`
  - `'product_listing'`
  - `'social_media_post'`
- `caption_length` (default: `'long'`): The length of the caption (`'any'`, `'very short'`, `'short'`, `'medium-length'`, `'long'`, `'very long'`, or an integer)
- `extra_options` (default: `[]`): List of extra options to include in the caption
- `name_input` (default: `''`): Name to use when referring to people/characters in the image
- `batch_size` (default: `4`): The number of images to process in parallel

### Output properties

- `image.{target_prop}`: The caption generated for the image

### Example

```python
dataset >> JoyCaptionAlphaTwo(
    caption_type='art_critic',
    caption_length='very long',
    extra_options=['Include information about lighting'],
    name_input='Alice'
)
```

## LlamaCaption

Generates captions for images using `meta-llama/llama-3.2-11B-Vision-Instruct`.

In order to download this model, you need to be logged into huggingface.

### Parameters

- `target_prop` (default: `'caption'`): The property to store the caption in
- `prompt` (default: `None`): The prompt to use for the Llama model (read the code)
- `instructions` (default: `None`): Additional instructions to include in the prompt
- `batch_size` (default: `1`): The number of images to process in parallel. If you are running out of memory, try reducing this value.

### Output properties

- `image.{target_prop}`: The caption generated for the image

### Example

```python
dataset >> LlamaCaption
```

## LLMCaptionVariations

Generates variations of existing image captions using LLaMA 3.1 8B Instruct. This preserves the original caption and adds numbered variations as new properties.

### Parameters

- `target_prop` (default: `'caption'`): Base property name to store variations in. Will append _1, _2 etc.
- `variations` (default: `2`): Number of variations to generate per image
- `parallel` (default: `20`): The number of images to process in parallel
- `temperature` (default: `0.7`): The temperature to use when sampling from the model
- `model` (default: `'together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'`): The LLM model to use for generating variations

### Output properties

- `image.{target_prop}`: The original caption (preserved)
- `image.{target_prop}_1`: First variation
- `image.{target_prop}_2`: Second variation (if variations > 1)
- etc.

### Example

```python
# Generate 3 variations of each caption
dataset >> LLMCaptionVariations(variations=3)
```

## LLMCaptionTransform

Computes image captions using a Large Language Model. This can be used for a number of purposes included:

- Cleaning up captions from VLMs
- Computing better captions by combining results from multiple VLMs
- Computing captions based on tags or other metadata
- Combining tags or other metadata with VLM caption results
- Transforming captions into different styles (e.g. fluid vs booru).

And so on. This is commonly one of the last nodes in a workflow, that determines the final caption.

Language models are accessed via API using [litellm](https://github.com/BerriAI/litellm). Model strings 
should follow their conventions. For example:

- `together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`
- `together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`
- `openai/gpt-4o`
- `openai/gpt-4o-mini`

Depending on the platform used, you will need to set environment variables like `OPENAI_API_KEY` or `TOGETHER_API_KEY` or `ANTHROPIC_API_KEY` to access the models.

Be aware that OpenAI and Anthropic models are relatively censored and may refuse certain tasks based on the content. We have had
good experiences with Llama and Qwen models.

LLM results are cached with the model, prompt, and parameters as the key, beprepared will never execute the same request twice.

We plan to support local LLMs in the future, and would welcome pull requests that implement this efficiently. 

### Parameters

- `model`: The name of the language model to use
- `prompt`: A function that takes an image and returns a prompt for the language model
- `target_prop` (default: `caption`): The property to store the transformed caption in
- `parallel` (default: `20`): The number of images to process in parallel. Adjust based on rate limits
- `temperature` (default: `0.5`): The temperature to use when sampling from the language model. In general, lower temperatures give more consistent and "safe" results and reduce the language model's tendency to hallucinate.

### Output properties

- `image.{target_prop}`: The transformed caption

### Example

```python

            dataset 
            >> JoyCaptionAlphaOne(target_prop='joycaption')
            >> GPT4oCaption(target_prop='gpt4ocaption')
            >> XGenMMCaption(target_prop='xgenmmcaption')
            >> QwenVLCaption(target_prop='qwenvlcaption')
            >> LlamaCaption(target_prop='llamacaption')
            >> Gemma3Caption(target_prop='gemma3caption')
            >> LLMCaptionTransform('together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
                                   lambda image: f"""
    Multiple VLMs have captioned this image. These are their results: 

    - JoyCaption: {image.joycaption.value}
    - GPT4oCaption: {image.gpt4ocaption.value}
    - XGenMMCaption: {image.xgenmmcaption.value}
    - QwenVLCaption: {image.qwenvlcaption.value}
    - LlamaCaption: {image.llamacaption.value}
    - Gemma3: {image.gemma3caption.value}

    Please generate a final caption for this image based on the above information. Your response should be the caption, with no extra text or boilerplate.
                                   """.strip(),
                                   target_prop='caption')
```


## QwenVLCaption 

Generates captions for images using `Qwen/Qwen2-VL-7B-Instruct`.

### Parameters

- `target_prop` (default: `'caption'`): The property to store the caption in
- `prompt` (default: `None`): The prompt to use for the Qwen 2 VL 7B model (read the code)
- `instructions` (default: `None`): Additional instructions to include in the prompt
- `batch_size` (default: `1`): The number of images to process in parallel. If you are running out of memory, try reducing this value.

### Output properties

- `image.{target_prop}`: The caption generated for the image

### Example

```python
dataset >> QwenVLCaption
```

## Gemma3Caption

Generates captions for images using Google's Gemma 3 12B Instruction-Tuned Vision-Language Model.

Gemma 3 is a powerful multimodal model that can process both text and images, generating high-quality text outputs. It supports a 128K token context window and is multilingual, supporting over 140 languages. The model excels at detailed image descriptions, visual question answering, and image analysis.

**Requirements:**
- **Transformers version:** Requires `transformers >= 4.46.0`. Run `pip install --upgrade transformers` if you encounter model loading errors.
- **VRAM:** Gemma 3 12B requires significant VRAM (24GB+ recommended). Consider using batch_size=1 to manage memory usage.

### Parameters

- `target_prop` (default: `'caption'`): The property to store the caption in
- `prompt` (default: Detailed description prompt): The prompt to use for image captioning
- `system_prompt` (default: `None`): System prompt to set the assistant's behavior
- `instructions` (default: `None`): Additional instructions to append to the prompt
- `batch_size` (default: `1`): The number of images to process in parallel. Keep at 1 for 12B model.

### Output properties

- `image.{target_prop}`: The caption generated for the image

### Example

```python
# Basic usage with default detailed captioning
dataset >> Gemma3Caption()

# Custom prompt for specific focus
dataset >> Gemma3Caption(
    prompt='What objects are visible in this image?'
)

# With system prompt and instructions
dataset >> Gemma3Caption(
    system_prompt='You are an art critic analyzing paintings.',
    prompt='Analyze this artwork',
    instructions='Focus on style, technique, and emotional impact'
)
```

## Passthrough

A node that does nothing and returns the dataset unchanged. This can be useful as a no-op placeholder in conditional pipeline branches.

### Example

```python
# Use Passthrough as a no-op alternative in a conditional
dataset >> (ProcessNode() if condition else Passthrough())
```

## MapCaption

Maps the current caption to a new caption using a function.

### Parameters

- `func`: Function that takes the current caption string and returns a new caption string.

### Output Properties

- `image.caption`: The transformed caption.

### Example

```python
# Add an exclamation mark to all captions
dataset >> MapCaption(lambda caption: caption + "!")

# Prepend a prefix to all captions
dataset >> MapCaption(lambda caption: f"A photo showing {caption}")
```

## SetCaption

Sets the `caption` property on an image.

### Parameters

- `caption`: The caption to set.

### Output Properties

- `image.caption`: The caption of the image.

### Example

```python
dataset >> SetCaption("ohwx person")
```

## XGenMMCaption

Generates captions for images using `Salesforce/xgen-mm-phi3-mini-instruct-r-v1`.

### Parameters

- `target_prop` (default: `'caption'`): The property to store the caption in
- `prompt` (default: `None`): The prompt to use for the xGen-mm model (read the code)
- `instructions` (default: `None`): Additional instructions to include in the prompt
- `batch_size` (default: `4`): The number of images to process in parallel. If you are running out of memory, try reducing this value.

### Output properties

- `image.{target_prop}`: The caption generated for the image

### Example

```python
dataset >> XGenMMCaption
```

## Florence2Caption

Generates captions for images using the Florence-2-large model.

### Parameters

- `target_prop` (default: `'caption'`): The property to store the caption in
- `task` (default: `Florence2Task.MORE_DETAILED_CAPTION`): The captioning task to perform. One of:
  - `Florence2Task.CAPTION`: Basic caption
  - `Florence2Task.DETAILED_CAPTION`: Detailed caption
  - `Florence2Task.MORE_DETAILED_CAPTION`: More detailed caption
- `batch_size` (default: `8`): The number of images to process in parallel. If you are running out of memory, try reducing this value.

### Output properties

- `image.{target_prop}`: The caption generated for the image

### Example

```python
dataset >> Florence2Caption(task=Florence2Task.DETAILED_CAPTION)
```

