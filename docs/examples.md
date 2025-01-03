# Examples

## How to run

To run the examples, save them as Python files (e.g. `workflow.py`) and run them using:

```bash
beprepared run workflow.py
```

You will almost always use a `Load` node to import images and a `Save` node to write out the cleaned data set. What comes in 
between can be simple or complex.

When you invoke a `Human*` node like `HumanFilter` or `HumanTag`, beprepared will launch a web-based interface on port 8989. 
If there are un-filtered or un-tagged images, you will be prompted to go to the web interface and perform filtering.

By convention, `Save` will place output images in the `output/` directory. For each image, there will be a companion `.txt` file
containing that image's caption. There will also be a `.json` file which contains all of the image's metadata. Finally, there is an
`index.html` which allows you to view the images and their metadata in a web browser.


## Captioning with a trigger word

This is a simple example of how to use beprepared to caption images based on a trigger word.

    (
        Load("/path/to/photos_of_me") 
        >> FilterBySize(min_edge=512)   
        >> ConvertFormat("JPEG")
        >> Dedupe
        >> SetCaption("ohwx person")
        >> Save
    )

## Auto-captioning using JoyCaption

    (
        Load("/path/to/photos_of_me")
        >> FilterBySize(min_edge=512)
        >> ConvertFormat("JPEG")
        >> JoyCaptionAlphaOne
        >> Save
    )

## Fuzzy deduplication

This example shows how to use FuzzyDedupe to remove duplicate images based on CLIP embeddings.

    (
        Load("/path/to/photos_of_me")
        >> ClipEmbed
        >> FuzzyDedupe
        >> Save
    )

## Filtering based on Aesthetic Score

This example shows how to use select the best 100 albums based on their aesthetic score.

    (
        Load("/path/to/photos_of_me")
        >> AestheticScore
        >> Sorted(lambda image: image.aesthetic_score.value, reverse=True)
        >> Take(100)
        >> Save
    )

## Filtering out NSFW content

This example shows how to filter NSFW content using NudeNet

    (
        Load("/path/to/images")
        >> NudeNet
        >> Filter(lambda image: not image.has_nudity)
        >> Save
    )

## Filter images manually then caption with GPT4o

To run this example, you will need to set `OPENAI_API_KEY` in your environment.

    from beprepared import *

    (
        Load("/path/to/photos_of_me")
        >> FilterBySize(min_edge=512)
        >> ConvertFormat("JPEG")
        >> HumanFilter
        >> GPT4oCaption
        >> Save
    )

## Manually tag images

    (
        Load("/path/to/photos_of_dogs")
        >> FilterBySize(min_edge=512)
        >> ConvertFormat("JPEG")
        >> HumanFilter
        >> HumanTag(tags=["labrador", "golden retriever", "poodle"])
        >> Apply(lambda image: image.caption.value = ', '.join(['dog'] + image.tags.value))
        >> Save
    )

## Captioning a mix of SFW and NSFW content using various VLMs

Some VLMs are more NSFW-friendly than others. This workflow shows how to split the workflow and use different
caption strategies for NSFW content.

    all = (
        Load("/path/to/images")
        >> FilterBySize(min_edge=512)
        >> ConvertFormat("JPEG")
        >> NudeNet
    )

    with_nudity    = all >> Filter(lambda image: image.has_nudity) >> JoyCaptionAlphaOne
    without_nudity = all >> Filter(lambda image: not image.has_nudity) >> GPT4oCaption

    Concat(with_nudity, without_nudity) >> Save

## Captioning using multiple VLMs 

This workflow shows how to use multiple VLMs to caption an image, and then combine the results into a single caption using `LLMCaptionTransform`.

To run this example, you will need to set `OPENAI_API_KEY` and `TOGETHER_AI_API_KEY` in your environment. 

LLM APIs are accessed using [litellm](https://github.com/BerriAI/litellm), and any model string supported by `litellm` should work here.

    from beprepared import *

    (
        Load("/path/to/images")
        >> FilterBySize(min_edge=512)
        >> ConvertFormat("JPEG")
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
        >> Save
    )

