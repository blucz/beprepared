# Scaling

## Downscale

Downscale images based on edge length constraints. The node supports two mutually exclusive modes:
- `max_edge`: Downscales images only if their largest edge exceeds the specified value
- `min_edge`: Downscales images so their smallest edge equals exactly the specified value

Images that don't meet the criteria for downscaling are left unchanged.

`DownscaleMethod.PIL` uses the `LANCZOS` resampling filter.

### Parameters

- `method` (default=`DownscaleMethod.PIL`): The method to use for downscaling (currently only `DownscaleMethod.PIL` is supported)
- `max_edge` (optional): The maximum edge length - scales down only if image's largest edge is larger
- `min_edge` (optional): The minimum edge length - scales down to make smallest edge exactly this size
- `format` (default=`'PNG'`): The format to save the downscaled images in (e.g., `'PNG'`, `'JPEG'`, `'WEBP'`)

**Note:** You must specify exactly one of `max_edge` or `min_edge`. They are mutually exclusive.

### Output properties

- `image.width`: The width of the image after downscaling
- `image.height`: The height of the image after downscaling
- `image.format`: The format of the image after downscaling
- `image.objectid`: The object ID of the image after downscaling
- `image.downscale_info`: Dictionary containing scaling information:
  - `method`: The scaling method used
  - `max_edge` or `min_edge`: The edge constraint value
  - `original_width`: Width before scaling
  - `original_height`: Height before scaling
  - `scaled_width`: Width after scaling
  - `scaled_height`: Height after scaling

### Examples

```python
# Downscale images so that the max edge of any image is 1024px
dataset >> Downscale(max_edge=1024)

# Downscale images so that the minimum edge is exactly 512px
dataset >> Downscale(min_edge=512)

# Downscale to minimum edge of 768px and save as JPEG
dataset >> Downscale(min_edge=768, format='JPEG')
```

### Behavior Details

**With `max_edge`:**
- If the largest dimension is already ≤ max_edge, the image is unchanged
- Otherwise, scales down proportionally so the largest dimension equals max_edge

**With `min_edge`:**
- If the smallest dimension is already ≤ min_edge, the image is unchanged
- Otherwise, scales down proportionally so the smallest dimension equals min_edge
- Useful for ensuring consistent minimum resolutions after cropping

## Upscale

Upsample images where their longest edge is less than `min_edge`. Images that are already large enough are not modified.

`UpscaleMethod.PIL` uses the `LANCZOS` resampling filter.
`UpscaleMethod.SWINIR` uses the SwinIR neural network model for high-quality upscaling.

_NOTE: ESRGAN is partially implemented, but it requires hacks to work because of [bugs](https://github.com/XPixelGroup/BasicSR/issues/533) in `basicsr`. There is a plan to work around this on our side, but it is not done yet._

## Parameters

- `method` (default=`UpscaleMethod.PIL`): The method to use for upscaling:
  - `UpscaleMethod.PIL`: Fast CPU-based upscaling using LANCZOS filter
  - `UpscaleMethod.SWINIR`: High-quality GPU-based upscaling using SwinIR neural network
- `min_edge` (default=`1024`): The minimum edge length for the upscaling
- `format` (default=`'PNG'`): The format to save the upscaled images in (e.g., `'PNG'`, `'JPEG'`, `'WEBP'`)

### Output properties

- `image.width`: The width of the image after upscaling
- `image.height`: The height of the image after upscaling
- `image.format`: The format of the image after upscaling
- `image.objectid`: The object ID of the image after upscaling

### Example

```python
# Upscale images so that the min edge of any image is 1024px
dataset >> Upscale(min_edge=1024)
```