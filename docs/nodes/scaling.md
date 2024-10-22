# Scaling

## Downscale

Downsample images where their longest edge is greater than `max_edge`. Images that are already small enough are not modified.

`UpscaleMethod.PIL` uses the `LANCOZ` resampling filter.

### Parameters

- `method` (default=`DownscaleMethod.PIL`: The method to use for downscaling (currently only `DownscaleMethod.PIL` is supported)
- `max_edge` (default=`1024`): The maximum edge length for the downscaling
- `format` (default=`'PNG'`): The format to save the downscaled images in (e.g., `'PNG'`, `'JPEG'`, `'WEBP'`)

### Output properties

- `image.width`: The width of the image after downscaling
- `image.height`: The height of the image after downscaling
- `image.format`: The format of the image after downscaling
- `image.objectid`: The object ID of the image after downscaling

### Example

```python
# Downscale images so that the max edge of any image is 1024px
dataset >> Downscale(max_edge=1024)
```

## Upscale

Upsample images where their longest edge is less than `min_edge`. Images that are already large enough are not modified.

`UpscaleMethod.PIL` uses the `LANCOZ` resampling filter.

_NOTE: AI based upscaling is planned for the future. ESRGAN is partially implemented, but it requires hacks to work 
because of [bugs](https://github.com/XPixelGroup/BasicSR/issues/533) in `basicsr`. There is a plan to work around this 
on our side, but it is not done yet._

## Parameters

- `method` (default=`UpscaleMethod.PIL`): The method to use for upscaling (currently only `UpscaleMethod.PIL` is supported)
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
