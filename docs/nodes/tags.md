# Tagging

Tags are a way to label images with metadata. They can be used for a number of different purposes:

- To track images through the pipeline 
- To filter images based on their tags
- To capture human tagging effort, and use it to generate captions using `LLMCaption`
- To capture human tagging effort, and consume it in another tool from the `image.json` sidecars. 

## AddTags

Adds tags to all images in a dataset

### Parameters

- `tags`: A list of tags to add to each image

### Output properties

- `image.tags`: The tags of the image, after adding the new tags

### Example

```python
# Single tag
dataset >> AddTags("labrador")

# Multiple tags
dataset >> AddTags(["labrador", "poodle", "labradoodle"])
```

```
    '''HumanTag is a node that allows you to tag images using a web interface.
      
       The domain is used to separate different sets of tags. For example, you could have a domain for tags 
       related to style and a domain for tags related to content. If you want to keep them separate, use 
       different domains. Likewise, if you have multiple sets of images that use different tagging practices, 
       you will want to use different domains.

       If the tag set evolves and you want to evaluate images again without losing work, increment the 
       version number. This will cause the web interface to re-evaluate all images.
    '''
    def __init__(self, 
                 domain: str = 'default', 
                 filter_domain: str = 'default', 
                 version = 1, 
                 tags: List[str] | List[List[str]] = [],
                 target_prop: str = 'tags',
                 skip_ui=False):
```
## HumanTag

Displays an efficient web interface that a human can use to tag images. Once you enter the `HumanTag` step you will be prompted to open a web browser and view the web interface.

Tagging effort is cached, and `HumanTag` only presents un-tagged images. If all images have been tagged, the web interface is skipped.

Tags are cached in a `domain`. This allows you to have unrelated sets of tags that are built and maintained separately, which can be useful for multi-concept tagging or
merging different datasets or subsets of your dataset differently. In most cases the `default` domain is sufficient.

By default, the Web UI only presents untagged images. If you want to re-present all images for tagging, you can increment `version` and it will re-present all images.

In the HumanTag UI, you can filter images by tapping the trash icon or pressing the down-arrow key. This is useful if you encounter into an undesirable image during tagging, or if you want to tag+filter in one step to avoid visiting each image twice.

### Parameters

- `domain` (default: `"default"`): The domain to use for the tags. 
- `version` (default: `1`): The version of the tags
- `tags` (default: `[]`): A list of tags to add to each image. This can be a flat list of strings, or a list of lists of strings. In the second case, each list is organized on a new row in the UI.
- `target_prop` (default: `tags`): The property name to store the tags in
- `skip_ui` (default: `False`): If `True`, the UI will not be displayed, and some images may be untagged. This is useful as a temporary measure when you want to debug further steps on a dataset without tagging all images.

### Output properties

- `image.{target_prop}`: The tags of the image, after adding the new tags

### Example

```python
# Flat list of tags
dataset >> HumanTag(tags=["labrador", "poodle", "labradoodle"])

# List of tags with layout for the UI
dataset >> HumanTag(tags=[
    ["Greyhound", "Whippet", "Afghan Hound"],                             # Sighthounds
    ["Rottweiler", "Siberian Husky", "Saint Bernard"],                    # Working Dogs
    ["Jack Russell Terrier", "Scottish Terrier", "Bull Terrier"]          # Terriers
    ["Labrador Retriever", "Golden Retriever", "Flat-Coated Retriever"])  # Retrievers
])

```

## RemoveTags

Removes tags from all images in a dataset.

### Parameters

- `tags`: A list of tags to remove from each image

### Output properties

- `image.tags`: The tags of the image, after removing the specified tags

### Example

```python
# Single tag
dataset >> RemoveTags("labrador")

# Multiple tags
dataset >> RemoveTags(["labrador", "poodle", "labradoodle"])
```

## RewriteTags 

Rewrites tags for all images in a dataset.

### Parameters

- `mapping`: A dictionary mapping old tag names to new tag names

### Output properties

- `image.tags`: The tags of the image, after rewriting the tags

### Example

```python
dataset >> RewriteTags({
    "outdoors": "outside",
    "indoors": "inside"
})
```





