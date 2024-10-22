# CLIP

## ClipEmbed 

Computes a CLIP embedding for each image in the dataset using the `openai/clip-vit-large-patch14` model.

CLIP embedding is quite fast and well worth it, as it unlocks a lot of functionality.

This node computes CLIP embeddings efficiently in batch, and supports multi-GPU operation. Several other 
nodes depend on CLIP embeddings, for example `FuzzyDedupe` and `SmartHumanFilter`. 

You may also use beprepared to compute CLIP embeddings for your own purposes, which can be consumed out 
of the `image.json` sidecars in the output directory.

### Parameters

- `batch_size` (default: 128): The number of images to process in parallel.
- `target_property` (default: "clip"): The property to store the CLIP embedding in.

### Output properties

- `image.{target_property}`: The CLIP embedding for the image.

### Example

```python
dataset >> ClipEmbed
```

