<p align="center">
  <img align="center" src="beprepared.jpg" width="460px" />
</p>
<p align="left">

## Introduction

beprepared is an easy and efficient way to prepare high quality image datasets for diffusion model fine-tuning.

It is designed to facilitate both human and machine-driven data preparation work in a non-destructive environment
that aggressively avoids duplication of effort, even as the data preparation workflow evolves. 

A typical data preparation workflow might look like this:

- Scrape images from the web
- Filter out images that are too small, or low quality
- Manually select images that are relevant to your task
- Auto-caption images using GPT-4o, JoyCaption, LLaVa, BLIP3, or another VLLM
- Manually tag images with concepts or that are not understood by the automatic captioner
- Use an LLM prompt to join tags + captions into a final caption, perhaps introducing additional rules
- Perform caption augmentation by generating a few caption variations for each image
- Create a training directory with foo.jpg, foo.txt, bar.jpg, bar.txt, ...

While it's possible to do this manually with the help of a bunch of little python scripts that work on your .jpg 
and .txt files, it can be very cumbersome, especially as you iterate on the process or add new images. If 
you've done this a lot, you probably have graveyard of one-off scripts and datasets in various states of 
disarray for historical reasons, and it's becoming unmanageable.

Furthermore, existing user interfaces for tasks that require a human in the loop, such as filtering scraped images, 
or manually tagging images for concepts that don't appear in pre-training data, are clumsy at best. In most cases
they modify caption files directly, and the UIs are not built for speed or efficiency, often lacking basic comforts
like keyboard shortcuts, image preloading, or mobile-friendly design.

Humans are human. If the tooling is bad, the datasets will be bad. If the datasets are bad, the models trained with 
them will be bad. If the models are bad, the consumers of those models will make bad images, or get frustrated, or 
give up. beprepared is designed to break this cycle by making the data preparation process as friendly as possible.

With beprepared, you can define a simple workflow like this:

      from beprepared import *

      with Workspace("mydataset") as workspace:
          (
              Load("/path/to/dog_photos")
              >> FilterBySize(min_edge=512)
              >> HumanFilter
              >> ConvertFormat("JPEG")
              >> JoyCaptionAlphaOne(target_property='caption')
              >> HumanTag(tags=["labrador", "golden retriever", "poodle"])
              >> Save
          )
          workspace.run()

When this workflow is executed, beprepared will first walk the `/path/to/dog_photos` directory to discover images, 
then ingest them into the workspace. Next, we will hit the HumanSelection step, and it will launch a web based UI 
with a single-task user interface focused solely on filtering images. Once all images have been selected or rejected, 
it can move on to the next step.

Each step along the way is cached, on an image-by-image basis. So if you run the workflow again, it will not need to
re-present the web interface for selection, nor will it need to re-caption images that have already been captioned. 
During human-in-the-loop steps, changes are committed to the database on every click, so you will _never_ lose work.

If you add new images to the source directory and run it again, only new images will be processed, using cached values
for the others. If you change the workflow, the system will preserve as much work as possible, avoiding expensive 
human-in-the-loop operations, or ML model invocations that have already been run.

Significantly more complex workflows are possible, and beprepared is designed to support them. See EXAMPLES.md for 
more examples.

## Limitations

This project is meant for using a single compute node to prepare data sets for fine-tuning diffusion models. 

It is not a goal of this project to prepare pre-training scale datasets with millions or billions of albums,
or manage distributed compute.

## Roadmap

beprepared is in a pre-alpha state. Much of the basic functionality is there, but it needs to be tested in real world
use cases, and functionality expanded as needed. The DSL and symbol names are not stable and are subject to change
at any time. 

Here are some items planned for the future:

- Implement a HumanRank node that uses pairwise rankings performed by a human to compute ELO ranks for images
- Implement nodes that train small NNs to predict tags, filter results, or rank results so that human work on a subset of a large dataset can be applied to the whole dataset
- Publish on PyPI and add installation instrucitons
- Improve performance of VLMs
- Support for multiple GPUs
- Improve error messages + "foolproof" nature of this software
- Improve experience on <24GB GPUs by automatically scaling batch sizes or using quantized models
- Write example documentation
- Write reference documentation for nodes
- Create guide for contributing new nodes
- Support data augmentation nodes like flip, caption variations, etc.
- Support sidecar .json files for better interoperability with other tools/code

In the future, beprepared will support LLM finetuning datasets as well, but that work will wait until image-related 
functionality is more mature.


