# Cropping Nodes

## CropToAspect

The `CropToAspect` node intelligently crops images to match specified aspect ratios. This is particularly useful for preparing datasets that need consistent aspect ratios for training or when targeting specific output formats.

### Features

- **Smart Aspect Matching**: Automatically selects the closest aspect ratio from your list to minimize cropping
- **Center Cropping**: Preserves the most important content by cropping from the center
- **Efficient Caching**: Processes each image only once, even across multiple workflow runs

### Parameters

- `aspect_ratios` (List[float]): List of target aspect ratios (width/height)
  - Example: `[0.5, 1.0, 1.5]` for portrait, square, and landscape
  - Example: `[1.0, 1.91, 0.8]` for common social media formats

### How It Works

1. For each image, the node calculates its original aspect ratio
2. Finds the closest matching ratio from your provided list
3. Calculates center crop dimensions to achieve the target ratio
4. Crops and saves the image with the new aspect ratio

### Usage Examples

#### Basic Usage
```python
from beprepared import *

(
    Load("raw_images")
    >> CropToAspect(aspect_ratios=[1.0])  # Crop all to squares
    >> Save("square_images")
)
```

#### Multiple Aspect Ratios
```python
# Prepare dataset with common aspect ratios
(
    Load("diverse_images")
    >> CropToAspect(
        aspect_ratios=[0.75, 1.0, 1.33]  # Portrait, square, landscape
    )
    >> Save("normalized_aspects")
)
```

#### Social Media Formats
```python
# Crop for various social media platforms
(
    Load("content")
    >> CropToAspect(
        aspect_ratios=[
            1.0,    # Instagram square
            0.8,    # Instagram portrait (4:5)
            1.91,   # Instagram landscape (1.91:1)
            0.5625, # Instagram story (9:16)
        ]
    )
    >> Save("social_ready")
)
```

#### Combined with Size Filtering
```python
# Full preprocessing pipeline
(
    Load("raw_dataset")
    >> FilterBySize(min_edge=1024)         # Remove small images first
    >> CropToAspect(
        aspect_ratios=[0.67, 1.0, 1.5]    # Common training ratios
    )
    >> FilterBySize(min_edge=512)          # Filter based on cropped size
    >> JoyCaptionAlphaOne                  # Caption the cropped images
    >> Save("training_ready")
)
```

### Properties Added

The node adds several properties to track the cropping operation:

- `crop_target_ratio`: The aspect ratio that was applied
- `crop_original_width`: Original image width before cropping
- `crop_original_height`: Original image height before cropping

These can be useful for debugging or further processing:

```python
(
    Load("images")
    >> CropToAspect(aspect_ratios=[1.0, 1.5])
    >> Filter(lambda img: img.crop_target_ratio.value == 1.0)  # Only squares
    >> Save("squares_only")
)
```

### Performance Considerations

- Images are cached after processing, so re-running workflows is fast
- The node loads images one at a time to minimize memory usage
- Consider filtering very small images first with `FilterBySize` to avoid unnecessary processing

### Common Use Cases

1. **Training Set Preparation**: Normalize aspect ratios for consistent batch processing
2. **Multi-Aspect Training**: Prepare datasets with specific aspect ratio buckets
3. **Platform-Specific Content**: Crop images for specific social media or display requirements
4. **Aspect Ratio Standardization**: Ensure all images match specific aspect ratios

### Tips

- Order aspect ratios from most to least common in your dataset for better matching
- Use `FilterBySize` before `CropToAspect` to pre-filter obviously unsuitable images
- Use `FilterBySize` after `CropToAspect` if you need to ensure minimum dimensions
- For maximum flexibility, include a wide range of aspect ratios
- The node always preserves the center of the image, so ensure important content is centrally located