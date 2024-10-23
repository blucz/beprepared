# Loading Images

## Load

Walks a directory of images recursively, loading them into the workspace.

During the load process, images are validated and attributes are extracted. Images are hardlinked into `_beprepared/objects` to track their contents properly while avoiding taking up additional disk space.

beprepared currently assumes that images are immutable. This is something we plan to work on, but for now, if you edit an image after processing in beprepared, consider changing the filename before re-processing that dataset so that beprepared picks up the changes.

### Parameters

- `dir`: The directory to walk for images

### Output properties

- `image.original_path`: The original path of the image
- `image.objectid`: The sha256 hash of the image
- `image.ext`: The extension of the image
- `image.width`: The width of the image
- `image.height`: The height of the image
- `image.format`: The format of the image, as determined by the `PIL` library (e.g. `JPEG`, `PNG`, `WEBP`, etc.)

### Example

```python
Load(.) >> ... other nodes ...
```
