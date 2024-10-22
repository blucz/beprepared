# Filtering

## FilterBySize

Excludes images from the dataset based on their size.

### Parameters

- `min_width` (default: `None`): The minimum width of the image
- `min_height` (default: `None`): The minimum height of the image
- `min_edge` (default: `None`): The minimum edge length of the image
- `max_width` (default: `None`): The maximum width of the image
- `max_height` (default: `None`): The maximum height of the image
- `max_edge` (default: `None`): The maximum edge length of the image

### Example

```python
dataset >> FilterBySize(min_width=768)
```

## Filter

The filter node is used to filter images based on a predicate.

### Parameters:

- `predicate`: A function that takes an image and returns a boolean.

### Examples:

```python
Filter(lambda image: image.aesthetic_score.value > 0.5)
```


