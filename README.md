<p align="center">
  <!-- this must be an absolute URL to work on PyPI -->
  <img align="center" src="https://raw.githubusercontent.com/blucz/beprepared/main/beprepared.jpg" width="460px" />
</p>
<p align="left">

beprepared is an easy and efficient way to prepare high quality image datasets for diffusion model fine-tuning.

It falicates both human and machine-driven data prep work in a non-destructive environment that aggressively 
avoids duplicated effort, even as your workflow evolves.

It is the most efficient way for one person to prepare image data sets with thousands of images or more.

## The Problem

A typical data prep workflow may look like this:

- Scrape images from the web
- Filter out images that are too small, or low quality
- Manually select images that are relevant to your task
- Auto-caption images using GPT-4o, JoyCaption, Llama 3.2, BLIP3, xGen-mm, or another VLM
- Manually tag images with information that is not understood by the VLM
- Use an LLM prompt to compose tags + VLM captions into a final caption, perhaps introducing additional rules
- Perform caption augmentation by generating a few caption variations for each image
- Create a training directory with `foo.jpg`, `foo.txt`, `bar.jpg`, `bar.txt`, ...

While it's possible to do this manually with the help of a bunch of python scripts that work on your .jpg 
and .txt files, it can be very cumbersome, especially as you iterate on the process or add new images. If 
you've done this a lot, you probably have graveyard of one-off scripts and datasets in various states of 
disarray.

Existing user interfaces for tasks that require a human in the loop, such as filtering scraped images, or 
manually tagging images are clumsy at best. In many cases they are built in sluggish frameworks like Gradio,
lack keyboard shortcuts, are not mobile friendly, and are not non-destructive. We don't need beautiful 
interfaces to work on image datasets, but we do need efficient interfaces to move quickly
through the work.

Humans are human. If the tooling is bad, the datasets will be bad. If the datasets are bad, the models trained with 
them will be bad. If the models are bad, the consumers of those models will make bad images, get frustrated, or 
give up. beprepared is designed to improve every stage of that process by making the data preparation process as 
efficient as possible.

## Command Line Interface

beprepared provides a powerful command-line interface for running workflows and managing your workspace:

```bash
# Install from PyPI
$ pip install beprepared

# Run a workflow file
$ beprepared run workflow.py

# Execute a quick operation
$ beprepared exec "Load('images') >> FilterBySize(min_edge=512) >> Info"
$ beprepared exec "Load('images') >> AestheticScore >> Sort >> Take(100) >> Save"

# Manage the workspace database
$ beprepared db list                    # List all cached properties
$ beprepared db list "aesthetic_score*" # List specific properties
$ beprepared db clear "clip_embed*"     # Clear specific cached data
```

The CLI has three main commands:

### `beprepared run` - Execute Workflow Files

Run complete workflow files that define your data preparation pipeline. For example:

```bash
$ beprepared run workflow.py
```

A typical workflow file looks like this:

    # workflow.py
    (
        Load("/path/to/dog_photos")
        >> FilterBySize(min_edge=512)    # Remove small images
        >> HumanFilter                   # Opens web UI for manual filtering
        >> ConvertFormat("JPEG")         # Standardize format
        >> JoyCaptionAlphaOne           # Auto-caption images
        >> HumanTag(tags=["labrador", "golden retriever", "poodle"])  # Manual tagging
        >> Save                          # Save to output/
    )

**[See More Examples](https://blucz.github.io/beprepared/examples)**

When you run this workflow, beprepared will:

1. Launch a web interface when human input is needed (filtering/tagging)
2. Cache all operations to avoid repeating work
3. Save the processed dataset to the output directory

Each step is cached, so if you run the workflow again:

- Previously filtered/tagged images won't need human review
- Previously captioned images won't need recaptioning
- Only new or modified images will be processed

When this workflow is executed, beprepared will first walk the `/path/to/dog_photos` directory to discover images, 
then ingest them into the workspace. Next, it will hit the HumanSelection step, and launch a web based UI 
with a single-task user interface focused solely on filtering images. Once all images have been filtered by a human, 
it can move on to the next step.

Each step along the way is cached, on an image-by-image basis. So if you run the workflow again, it will not need to
re-present the web interface for selection, nor will it need to re-caption images that have already been captioned. 
During human-in-the-loop steps, changes are committed to the database on every click, so you will _never_ lose work.

If you add new images to the source directory and run it again, only new images will be processed, using cached values
for the others. If you change the workflow, the system will preserve as much work as possible, avoiding expensive 
human-in-the-loop operations, or ML model invocations that have already been run.

Significantly more complex workflows are possible, and beprepared is designed to support them. See 
[the docs](https://blucz.github.io/beprepared/examples) for more examples.

`beprepared run` is the main way to use beprepared.

### `beprepared exec` - Quick Operations

Execute one-line operations without creating a workflow file:

```bash
$ beprepared exec "Load('raw_images') >> HumanFilter >> Save('filtered')"
$ beprepared exec "Load('dataset') >> JoyCaptionAlphaOne >> Save"
$ beprepared exec "Load('photos') >> NudeNet >> Info"
$ beprepared exec "Load('photos') >> EdgeWatermarkRemoval >> Save('photos_cleaned')"
```

### `beprepared db` - Manage the Workspace (advanced)

View and manage cached operations in your workspace:

```bash
$ beprepared db list                                 # List all properties
$ beprepared db list "llm*"                          # List specific properties
$ beprepared db clear                                # Clear all cached data
$ beprepared db clear "gemini*"                      # Clear specific cached data
$ beprepared db clear -d "mydomain" "humanfilter*"   # Clear specific cached data
```
This lets you inspect the cache and clear out items, for example if you want to force re-running a step for some reason. It's most commonly used while developing beprepared, but we can imagine other use cases. 

## Features

- Flexible workflow definitions using a Python based DSL
- Non-destructive workflow execution
- Caching of intermediate results to avoid duplicate work
- Automatic Captioning using JoyCaption, Llama 3.2, BLIP3, xGen-mm, GPT-4o, Gemini, Molmo, and Florence2
- Human-in-the-loop filtering and tagging
- Nudity detection using NudeNet
- Watermark removal using Florence2
- Improving captions using LLMs
- Upscaling and downscaling images using PIL
- Filtering images based on size
- Computing CLIP embeddings for images
- Generates JSON sidecar files for each image so that you can use the data in other tools or scripts
- Precise and Fuzzy (CLIP-based) image deduplication
- Aesthetic scoring
- Collection operations like Map, Apply, Filter, Sort, Shuffle, Concat, and Random Sampling
- Multi-GPU support for faster captioning, CLIP encoding, and watermark removal on multi-GPU systems

## Documentation

The full documentation is available at [https://blucz.github.io/beprepared](https://blucz.github.io/beprepared)

## Limitations

This project is used to prepare data sets for fine-tuning diffusion models on a single compute node. 

It is not a goal of this project to help with preparing pre-training datasets with millions or billions of images. 
That would require a fundamentally more complex distributed architecture and would make it more difficult
for the community to work with and improve this tool.

This project is currently developed on 24GB GPUs, and is not optimized for smaller GPUs. We welcome patches that make 
this software more friendly to smaller GPUs or faster on larger GPUs.

