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
- Manually tag images with concepts or that are not understood by the VLM
- Use an LLM prompt to compose tags + captions into a final caption, perhaps introducing additional rules
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

### run - Execute Workflow Files
Run complete workflow files that define your data preparation pipeline:
```bash
$ beprepared run workflow.py
```

### exec - Quick Operations
Execute one-line operations without creating a workflow file:
```bash
$ beprepared exec "Load('raw_images') >> HumanFilter >> Save('filtered')"
$ beprepared exec "Load('dataset') >> JoyCaptionAlphaOne >> Save"
$ beprepared exec "Load('photos') >> NudeNet >> Info"
```

### db - Manage the Workspace
View and manage cached operations in your workspace:
```bash
$ beprepared db list              # List all properties
$ beprepared db list "*caption*"  # Search for specific properties
$ beprepared db clear            # Clear all cached data
$ beprepared db clear "temp_*"   # Clear specific cached data
```

## Documentation

The full documentation is available at [https://blucz.github.io/beprepared](https://blucz.github.io/beprepared)

## Quick Example

Create a workflow file:

    # workflow.py
    (
        Load("/path/to/dog_photos")
        >> FilterBySize(min_edge=512)
        >> HumanFilter
        >> ConvertFormat("JPEG")
        >> JoyCaptionAlphaOne
        >> HumanTag(tags=["labrador", "golden retriever", "poodle"])
        >> Save
    )

Run it:

    $ beprepared run workflow.py

The workflow will:
1. Launch a web interface when human input is needed (filtering/tagging)
2. Cache all operations to avoid repeating work
3. Save the processed dataset to the output directory

**[See More Examples](https://blucz.github.io/beprepared/examples)**

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

## Features

- Flexible workflow definitions using a Python based DSL
- Non-destructive workflow execution
- Caching of intermediate results to avoid duplicate work
- Automatic Captioning using JoyCaption, Llama 3.2, BLIP3, xGen-mm, GPT-4o, Gemini, Molmo, and Florence2
- Human-in-the-loop filtering and tagging
- Nudity detection using NudeNet
- Improving captions using LLMs
- Upscaling and downscaling images using PIL
- Filtering images based on size
- Computing CLIP embeddings for images
- Generates JSON sidecar files for each image so that you can use the data in other tools or scripts
- Precise and Fuzzy (CLIP-based) image deduplication
- Aesthetic scoring
- Collection operations like Map, Apply, Filter, Sort, Shuffle, Concat, and Random Sampling

## Limitations

This project is used to prepare data sets for fine-tuning diffusion models on a single compute node. Currently it
supports only one GPU, but multi-GPU support is planned.

It is not a goal of this project to help people preparing pre-training datasets with millions or billions of images. 
That would require a fundamentally more complex distributed architecture and would make it more difficult
for the community to work with and improve this tool.

This project is currently developed on 24GB GPUs, and is not optimized for smaller GPUs. We welcome patches that make 
this software more friendly to smaller GPUs. Most likely this will involve tuning batch sizes or using quantized models.

