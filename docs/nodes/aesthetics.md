# Aesthetics

## AestheticScore 

Computes an aesthetic score for each image in the dataset using the simple aesthetics predictor model.

This is most commonly combined with `Sort` and `Filter` nodes in order to select images based on their aesthetic score.

### Parameters

- `batch_size` (default: 256): The number of images to process in parallel.

### Output properties

- `image.aesthetic_score`: The aesthetic score of the image.

### Example

```python
AestheticScore
```

