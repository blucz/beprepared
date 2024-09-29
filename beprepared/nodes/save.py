from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.utils import copy_or_hardlink
from typing import List

import shutil
import os

class Save(Node):
    def __init__(self, dir=None):
        super().__init__()
        self.dir = dir or "output"

    def eval(self, dataset) -> Dataset:
        dst_path = os.path.join(self.workspace.dir, self.dir)
        shutil.rmtree(dst_path, ignore_errors=True)

        for image in dataset.images:
            objectid       = image.objectid.value
            original_path  = image.original_path.value
            ext            = image.ext.value
            src_image_path = self.workspace.get_path(image)
            dst_image_path = os.path.join(dst_path, f"{os.path.basename(original_path)}_{objectid}{ext}")
                
            os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
            copy_or_hardlink(src_image_path, dst_image_path)

            if image.caption.has_value:
                dst_caption_path = os.path.join(dst_path, f"{os.path.basename(original_path)}_{objectid}.txt")
                with open(dst_caption_path, 'w') as f:
                    f.write(image.caption.value)

        self.log.info("saving html")
        generate_html(dst_path, dataset.images)
        return dataset


def generate_html(dst_path: str, images: List):
    if len(images) == 0: return

    html_path = os.path.join(dst_path, "dataset.html")
    with open(html_path, 'w') as f:
        f.write("<html><head><title>Dataset</title></head><body>\n")
        f.write("<table border='1' style='width:100%; border-collapse: collapse;'>\n")
        f.write("<tr><th>Image</th><th>Properties</th></tr>\n")
        for image in images:
            objectid      = image.objectid.value
            original_path = image.original_path.value
            ext           = image.ext.value
            image_filename = f"{os.path.basename(original_path)}_{objectid}{ext}"
            caption = image.caption.value if image.caption.has_value else ""
            f.write("<tr>\n")
            f.write(f"<td><img src='{image_filename}' style='max-width:300px;'></td>\n")
            f.write("<td style='padding: 8px'>\n")

            for k,v in image.props.items():
                if k.startswith('_') :
                    continue
                if not v.has_value: continue
                value = v.value
                if hasattr(value, 'show'):
                    s = f"<pre>{value.show()}</pre>"
                else:
                    s = repr(value)
                f.write(f"<p><strong>{k}</strong> {s}</p>\n")
            f.write("</td>\n")
            f.write("</tr>\n")
        f.write("</table>\n")
        f.write("</body></html>")
