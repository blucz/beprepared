# Filtering

## Filter

The filter node is used to filter images based on a predicate.

### Parameters

- `predicate`: A function that takes an image and returns a boolean.

### Example

```python
dataset >> Filter(lambda image: image.aesthetic_score.value > 0.5)
```

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

## HumanFilter

Displays an efficient web interface that a human can use to filter images. Once you enter the `HumanFilter` step you will be prompted to open a web browser and view the web interface.

Filter results are cached, `HumanFilter` only presents un-filtered images. If all images have been filtered, the web interface is skipped.

The web interface is designed to be comfortable on both desktop and mobile platforms. If you have a keyboard, you can efficiently move through the list of image using left/right arrow keys, accept an image using the up arrow, and reject using the down arrow. You can also use WASD keys in the same way if you prefer to filter images left-handed.

Filter results are cached in a `domain`. This allows you to have unrelated filter judgements that are built and maintained separately.

### Parameters

- `domain` (default: `"default"`): The domain to use for caching the filter results.

### Output properties

- `image.passed_human_filter`: The filter result of the image.

### Example

```python
dataset >> HumanFilter
```
