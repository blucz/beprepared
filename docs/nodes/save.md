# Saving Datasets

## Save

The `Save` node saves images and captions to a directory, following `foo.jpg`, `foo.txt`, `bar.jpg`, `bar.txt`, etc. 

Additionally, an `index.html` file is generated that can be used to browse the images and their properties for evaluation purposes.

Images are hardlinked to the `_beprepared/objects` directory, so while new files are created, they should not take up additional space on the disk.


### Parameters

- `dir` (default: `output`): The directory to save the images to. Relative paths are computed relative to the workspace directory.
- `captions` (default: `True`): Whether to save captions files next to image files
- `sidecars` (default: `True`): Whether to save .json files next to the image which contain image properties
- `caption_ext` (default: `.txt`): The extension to use for captions files

### Example

```python
dataset >> Save
```
