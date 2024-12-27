from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.utils import copy_or_hardlink
from typing import List, Dict
import numpy as np
import urllib.parse
import json
import shutil
import os
from collections import defaultdict

class Save(Node):
    '''Saves images and captions to a directory, following foo.jpg, foo.txt, bar.jpg, bar.txt, etc. Additionally, an `index.html` file is generated that displays the images and their properties for evaluation purposes.''' 
    def __init__(self, dir=None, captions=True, sidecars=True, caption_ext=".txt"):
        '''Initializes the Save node

        Args:
            dir (str): The directory to save the images to. Relative paths are computed relative to the workspace directory.
            captions (bool): Whether to save captions files next to image files (default is True)
            sidecars (bool): Whether to save .json files next to the image which contain image properties (default is True)
            caption_ext (str): The extension to use for captions files (default is ".txt", some software prefers ".caption")
        '''
        super().__init__()
        self.dir = dir or "output"
        self.captions = captions
        self.sidecars = sidecars
        self.caption_ext = caption_ext

    def eval(self, dataset) -> Dataset:
        # for relative path, we need to use os.path.join. For absolute, just use it 
        if not os.path.isabs(self.dir):
            dst_path = os.path.join(self.workspace.dir, self.dir)
        else:
            dst_path = self.dir
        shutil.rmtree(dst_path, ignore_errors=True)
        
        # Track used filenames for each objectid to handle duplicates
        used_names: Dict[str, int] = defaultdict(int)
        
        for image in dataset.images:
            objectid       = image.objectid.value
            original_path  = image.original_path.value
            ext           = image.ext.value
            src_image_path = self.workspace.get_path(image)
            base_filename  = os.path.basename(original_path)[:80]    # limit length to keep filesystems happy.
            
            # Get suffix for duplicate images
            suffix = ""
            count = used_names[f"{base_filename}_{objectid}"]
            if count > 0:
                suffix = f"_{count}"
            used_names[f"{base_filename}_{objectid}"] += 1
            
            # Create filenames with suffix
            dst_image_path = os.path.join(dst_path, f"{base_filename}_{objectid}{suffix}{ext}")
            
            os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
            copy_or_hardlink(src_image_path, dst_image_path)

            if self.captions:
                if image.caption.has_value and image.caption.value:
                    dst_caption_path = os.path.join(dst_path, f"{base_filename}_{objectid}{suffix}{self.caption_ext}")
                    with open(dst_caption_path, 'w') as f:
                        f.write(image.caption.value)

            if self.sidecars:
                sidecar_data = { }
                for k,v in image.props.items():
                    if k.startswith('_') or not v.has_value: continue
                    v = v.value
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif hasattr(v, 'to_json'):
                        v = v.to_json()
                    sidecar_data[k] = v
                sidecar_path = os.path.join(dst_path, f"{base_filename}_{objectid}{suffix}.json")
                with open(sidecar_path, 'w') as f:
                    json.dump(sidecar_data, f, indent=2)

        self.log.info("saving html")
        generate_html(dst_path, dataset.images, used_names)
        return dataset

def generate_html(dst_path: str, images: List, used_names: Dict[str, int]):
    if len(images) == 0: return

    html_path = os.path.join(dst_path, "index.html")
    with open(html_path, 'w') as f:
        f.write("<html><head><title>Dataset</title></head><body>\n")
        f.write("<table border='1' style='width:100%; border-collapse: collapse;'>\n")
        f.write("<tr><th>Image</th><th>Properties</th></tr>\n")
        for image in images:
            objectid      = image.objectid.value
            original_path = image.original_path.value
            ext           = image.ext.value
            base_filename = os.path.basename(original_path)[:80]    # limit length to keep filesystems happy.
            # Get suffix for HTML display
            suffix = ""
            count = used_names[f"{base_filename}_{objectid}"] - 1
            if count > 0:
                suffix = f"_{count}"
            image_filename = f"{base_filename}_{objectid}{suffix}{ext}"
            image_filename_encoded = urllib.parse.quote(image_filename)
            caption = image.caption.value if image.caption.has_value else ""
            f.write("<tr>\n")
            f.write(f"<td><img src='{image_filename_encoded}' loading='lazy' style='max-width:300px;'></td>\n")
            f.write("<td style='padding: 8px'>\n")

            for k,v in image.props.items():
                if k.startswith('_') :
                    continue
                if not v.has_value: continue
                value = v.value
                if isinstance(value, np.ndarray):
                    s = np.array2string(value, threshold=50)
                elif hasattr(value, 'show'):
                    s = f"<pre>{value.show()}</pre>"
                elif isinstance(value, list) and len(value) > 100:
                    tostr = lambda x: f'{x:.3f}' if isinstance(x, float) else str(x)
                    s = f"[{', '.join(tostr(x) for x in (value[:25] + ['...'] + value[-25:]))}]"
                else:
                    s = repr(value)
                f.write(f"<p><strong>{k}</strong> {s}</p>\n")
            f.write("</td>\n")
            f.write("</tr>\n")
        f.write("</table>\n")
        f.write("</body></html>")

__all__ = ['Save']
