# Watermark Removal

The EdgeWatermarkRemoval node uses Florence-2 to detect watermarks, logos, and text near image edges and automatically crop them out.

## Usage

```python
from beprepared.nodes import EdgeWatermarkRemoval

# Basic usage with default settings
dataset >> EdgeWatermarkRemoval() >> Save("output")

# Custom crop threshold
dataset >> EdgeWatermarkRemoval(max_crop_percent=0.05) >> Save("output")

# Preview mode - draws bounding boxes instead of cropping so you can see what will happen
dataset >> EdgeWatermarkRemoval(preview_crop=True) >> Save("output")
```

## Parameters

- `max_crop_percent` (float, default=0.15): Maximum proportion of width/height that can be cropped from any edge. Value between 0.0-1.0.
- `preview_crop` (bool, default=False): If True, draws bounding boxes and proposed crop region on the original image instead of actually cropping the image.

## How it Works

1. Uses Florence-2's caption-to-phrase-grounding to detect watermarks, logos and text in images
2. Filters detections to only consider those that:
   - Are completely within max_crop_percent (15%) of any edge
   - Come within 3% of an edge (to avoid false positives)
3. For valid watermarks, determines which edge would require the smallest crop to remove all watermarks
4. Crops that edge if a valid solution is found

## Example

```python
(
    Load("input_images") 
    >> EdgeWatermarkRemoval(max_crop_percent=0.15)
    >> Save("output_images")
)
```

This will:
1. Load images from "input_images" directory
2. Detect and remove watermarks near edges, cropping up to 15% from one edge if needed
3. Save the processed images to "output_images" directory
