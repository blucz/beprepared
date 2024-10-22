# Utils

## Filter

The filter node is used to filter images based on a predicate.

### Parameters

- `predicate`: A function that takes an image and returns a boolean.

### Example

```python
dataset >> Filter(lambda image: image.aesthetic_score.value > 0.5)
```

## Concat

The concat node is used to concatenate one or more datasets.

### Parameters

- `*nodes`: One or more nodes.

### Examples

```python
# Direct usage
Concat(Load("/path/to/photos_of_me"), Load("/path/to/photos_of_dogs"))

# With << syntax
Concat << Load("/path/to/photos_of_me") << Load("/path/to/photos_of_dogs")
```

## Info

The info node prints information about the images in a dataset to stdout. This is useful for debugging small datasets and eyeballing results.

For larger datasets, use the `index.html` file in the output directory next to the images. This contains the same info, but can be viewed in a web browser.

### Parameters

- `include_hidden_properties` (default: `False`): If `True`, hidden properties will be included in the output.

### Example

```python
dataset >> Info
```

## SetCaption

Sets the `caption` property on an image.

### Parameters

- `caption`: The caption to set.

### Output Properties

- `image.caption`: The caption of the image.

### Example

```python
dataset >> SetCaption("ohwx person")
```

## Take 

The take node is used to take a fixed number of images from a dataset. 

It can also be used for random sampling.

### Parameters

- `n`: The number of images to take.
- `random` (default: `False`): If `True`, images will be taken randomly.
- `seed` (default: `None`): The seed for the random number generator, in case you want to select the same random images repeatedly.

### Example

```python
# Take the first 10 images
dataset >> Take(10)

# Take 10 random images
dataset >> Take(10, random=True)

# Take 10 random images, but always the same ones
dataset >> Take(10, random=True, seed=42)
```

## Sorted

Sorts images in the dataset based on a `key` function.

### Parameters

- `key`: A function that takes an image and returns a value to sort by.
- `reverse` (default: `False`): If `True`, the images will be sorted in descending order.

### Example

```python
dataset >> Sorted(lambda image: image.aesthetic_score.value, reverse=True)
```

## Shuffle

Shuffles the images in the dataset.

### Example

```python
Shuffle
```

## Map

Maps a function over the images in a dataset.

### Parameters

- `fn`: A function that takes an image and returns a new image.

### Example

```python
dataset >> Map(lambda image: image.with_props(caption=image.caption.value.upper()))
```

## Apply

Applies a function to each image in a dataset.

### Parameters

- `fn`: A function that takes an image and returns nothing.

### Example

```python

dataset >> Apply(lambda image: image.caption.value = image.caption.value.upper())
```

## Set

Sets properties on an image.

### Parameters

- `**kwargs`: The properties to set.

### Output Properties

- `image.{key}`: Properties specified in `kwargs`

### Example

```python
dataset >> Set(tags=["person", "ohwx"], myprop=12345, caption="my caption")
```

## Fail 

The fail node is used to fail the workflow with an error message. This is useful for debugging and testing.

### Parameters

- `message` (default: `"error"`): The error message.

### Example

```python
dataset >> Fail("Something went wrong")
```

## Sleep

The sleep node is used to pause the workflow for a specified number of seconds. This is useful for testing and debugging.

### Parameters

- `seconds`: The number of seconds to sleep.

### Example

```python
Sleep(5)
```


