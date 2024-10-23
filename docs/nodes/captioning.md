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
            >> LLMCaptionTransform('together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
                                   lambda image: f"""
    Multiple VLMs have captioned this image. These are their results: 

    - JoyCaption: {image.joycaption.value}
    - GPT4oCaption: {image.gpt4ocaption.value}
    - XGenMMCaption: {image.xgenmmcaption.value}
    - QwenVLCaption: {image.qwenvlcaption.value}
    - LlamaCaption: {image.llamacaption.value}

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

