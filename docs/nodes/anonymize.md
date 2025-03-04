# Face Anonymization

## Anonymize

The `Anonymize` node detects and anonymizes faces in images using the CenterFace face detection model. This is useful for privacy protection when working with datasets containing people. It can also prevent overfitting if your dataset is unbalanced.

It is crucial that when you caption your images for lora training, you mention that faces have been blurred. If not, you are likely to cause the model to forget how to generate faces. You may also want to consider using regularization data alongside this technique.


### Methods

The node supports several anonymization methods:

- `AnonymizeMethod.BLUR`: Blurs detected faces (default)
- `AnonymizeMethod.SOLID`: Covers faces with solid black rectangles
- `AnonymizeMethod.MOSAIC`: Applies a mosaic/pixelation effect to faces
- `AnonymizeMethod.NONE`: Detects faces but doesn't modify them (useful for testing)

### Parameters

- `method` (default: `AnonymizeMethod.BLUR`): The anonymization method to use
- `threshold` (default: `0.4`): Detection confidence threshold (0.0-1.0)
- `mask_scale` (default: `1.0`): Scale factor for face masks to ensure complete coverage
- `ellipse` (default: `True`): Use elliptical masks instead of rectangular ones (only for BLUR method)
- `mosaic_size` (default: `20`): Size of mosaic blocks when using the MOSAIC method

### GPU Acceleration

The node automatically uses available CUDA GPUs for faster processing. If multiple GPUs are available, it will distribute the workload across them.

### Example

```python
# Basic usage with default settings (blur faces)
dataset >> Anonymize()

# Use solid black rectangles with higher detection threshold
dataset >> Anonymize(method=AnonymizeMethod.SOLID, threshold=0.5)

# Apply mosaic effect with larger blocks
dataset >> Anonymize(method=AnonymizeMethod.MOSAIC, mosaic_size=30)

# Increase mask size to cover more of the face area
dataset >> Anonymize(mask_scale=1.2)
```

### Notes

- Face detection works best on clear, front-facing faces
- Increasing `threshold` reduces false positives but may miss some faces
- Increasing `mask_scale` helps ensure complete face coverage
- The node caches results, so re-running with the same parameters is very fast
