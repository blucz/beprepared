# Deduplication

## ExactDedupe 

This deduplicates images based on their SHA256 hash. This is extremely fast and simple, but does not catch perceptually similar images. 

If that is a priority, try `FuzzyDedupe` instead, or better yet use `ExactDedupe >> FuzzyDedupe` in sequence.

### Example

```python
ExactDedupe
```

## FuzzyDedupe

Deduplicates images based on perceptual similarity. 

This process uses [clip embeddings](clip.md) and an ANN (approximate nearest neighbors) index to find groups of images that are within `threshold` cosine 
similarity of each other. You can monitor the clusters by setting `debug_html` to a path where an HTML file will be saved that shows the images 
in each cluster. Using this, you can tune the `threshold` parameter to get the desired deduplication results.

The n_trees and n_neighbors parameters control the accuracy and speed of the ANN index. Higher values will be more accurate but slower. The 
default values are good for most cases.

### Parameters

- `threshold` (default: 0.95): The cosine similarity threshold for images to be considered duplicates.
- `debug_html` (default: 'fuzzy_dedupe.html'): If set, an HTML file will be saved with the images in each cluster for quality monitoring.
- `n_trees` (default: 10): The number of trees to build in the ANN index.
- `n_neighbors` (default: 50): The number of neighbors to search for in the ANN index.

### Example

```python
# Basic usage showing Clip Encoder and Fuzzy Dedupe
dataset >> ClipEncode >> FuzzyDedupe

# Advanced usage for tuning parameters
dataset >> ClipEncode >> FuzzyDedupe(threshold=0.9, debug_html='fuzzy_dedupe.html')

# Adjusting ANN parameters
dataset >> ClipEncode >> FuzzyDedupe(n_trees=20, n_neighbors=100)
```

