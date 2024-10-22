# Format Conversion

## ConvertFormat

Converts images to a specified format. Images already in that format will not be modified. 

The most common use cases are converting to `PNG` or `JPEG` for compatibility with training pipelines 
that do not support `WEBP` or other formats.

### Parameters

- `format`: The format to convert to as a PIL format string, e.g. `'JPEG'`, `'PNG'`, `'WEBP'`

### Output properties

- `image.format`: The format of the image, after conversion, e.g. `'PNG'`

### Example

```python
dataset >> ConvertFormat('PNG')
```
