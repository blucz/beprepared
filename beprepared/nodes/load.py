import os
import time
from io import BytesIO
from PIL import Image as PILImage, UnidentifiedImageError

from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.image import Image
from beprepared.workspace import Workspace
from beprepared.properties import CachedProperty, ConstProperty
from beprepared.utils import shorten_path

def _find_images(directory):
    valid_extensions = {'.jpg', '.png', '.webp'}
    
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() in valid_extensions:
                yield os.path.join(dirpath, filename)

class Load(Node):
    '''Loads images from a directory. Images are walked on each workflow run, so new images can be added to the directory and picked up on the next run.

    This node assumes that images are immutable once they are in the workspace. If an image is modified after having its sha256 cached, we will not pick up the change.

    If you plan to modify images, change their filenames. We may add an option in the future to pick up changes automatically, but it could have significant performance 
    impact, especially for large datasets. Since most images are scraped from the web or generated, we think this is a reasonable tradeoff.'''
    def __init__(self, dir: str):
        '''Initializes the Load node

        Args:
            dir (str): The directory to load images from
        '''
        super().__init__()
        self.dir = dir

    def eval(self):
        # TODO: If an image is modified after having its sha256 cached, we never pick up the 
        #       change. We should probably be doing something with file timestamps to guard against this.
        #       Note that this could have potentially annoying effects, especially for the Human* nodes,
        #       wherein a human might have to go do a bunch of little steps just because they cleaned
        #       something up in photoshop. This may requires some thought, or maybe we just want to 
        #       explicitly require that images are immutable once they're in the workspace.
        dataset = Dataset()
        ws = Workspace.current
        skipped = 0
        import_count = 0
        total_processed = 0
        start_time = time.time()
        for path in _find_images(self.dir):
            try:
                original_path_prop = ConstProperty(path)
                objectid_prop      = CachedProperty('objectid', path)
                did_import = False
                if not objectid_prop.has_value:
                    bytes = open(path, 'rb').read()
                    try:
                        with PILImage.open(BytesIO(bytes)) as pil_image:
                            if pil_image.format not in Image.ALLOWED_FORMATS:
                                self.log.warning(f"Skipping {shorten_path(path)} because it is not in the allowed formats")
                                skipped += 1
                                continue
                            pil_image.verify()
                    except:
                        self.log.warning(f"Skipping {shorten_path(path)} because it is not a valid image")
                        skipped += 1
                        continue
                    objectid_prop.value = ws.db.put_object(path)
                    did_import = True
                    import_count += 1
                height_prop        = CachedProperty('height', objectid_prop.value)
                width_prop         = CachedProperty('width',  objectid_prop.value)
                format_prop        = CachedProperty('format', objectid_prop.value)
                if not height_prop.has_value or not width_prop.has_value:
                    pil_image = PILImage.open(path)
                    height_prop.value = pil_image.height
                    width_prop.value = pil_image.width
                    format_prop.value = pil_image.format
                ext_prop           = ConstProperty(os.path.splitext(path)[1])
                image              = Image(
                    original_path=original_path_prop, 
                    objectid=objectid_prop, 
                    ext=ext_prop,
                    width=width_prop,
                    height=height_prop,
                    format=format_prop
                )
                #if did_import:
                #    self.log.info(f"Imported {shorten_path(path)} ({image.width.value}x{image.height.value} {image.format.value})")
                dataset.images.append(image)
                total_processed += 1
                if total_processed % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed
                    self.log.info(f"Progress: processed {total_processed} images ({import_count} imported) at {rate:.1f} images/sec")
            except:
                self.log.exception(f"Error loading {shorten_path(path)}")
                skipped += 1
        self.log.info(f"Loaded {len(dataset.images)} images, imported {import_count}, skipped {skipped}")
        return dataset
