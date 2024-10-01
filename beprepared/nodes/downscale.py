from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.properties import CachedProperty, ConstProperty
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

class DownscaleMethod:
    PIL    = "PIL"

class Downscale(Node):
    def __init__(self, method=DownscaleMethod.PIL, max_edge=1024, format='PNG'):
        super().__init__()
        self.max_edge = max_edge
        self.method   = method 
        self.format   = format

    def downscale_image_pil(self, image, max_edge):
        # Load the image using Pillow
        image_path = self.workspace.get_path(image)
        image = Image.open(image_path)
        
        # Resize so that the shorter side is max_edge
        width, height = image.size
        if width > height:
            new_width = max_edge
            new_height = int((max_edge / width) * height)
        else:
            new_height = max_edge
            new_width = int((max_edge / height) * width)

        self.log.debug(f"Downscaling {image_path} using PIL.")
        self.log.debug(f"Original size: {width}x{height}, new size: {new_width}x{new_height}")

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        if resized_image.mode != 'RGB':
            resized_image = resized_image.convert('RGB')

        byte_array = BytesIO()
        image.save(byte_array, format=self.format)
        objectid = self.workspace.put_object(byte_array.getvalue())

        return {
            'width': new_width,
            'height': new_height,
            'objectid': objectid
        }

    def eval(self, dataset) -> Dataset:
        toconvert = []
        mapping = { x: x for x in dataset.images }

        def newimage(image): 
            data = image._downscale_data.value
            width = data['width']
            height = data['height']
            objectid = data['objectid']
            return image.with_props({
                'width':  ConstProperty(width),
                'height': ConstProperty(height),
                'format': ConstProperty(self.format),
                'downscale_info': ConstProperty({
                    'method': self.method,
                    'max_edge': self.max_edge,
                    'original_width': image.width.value,
                    'original_height': image.height.value,
                    'scaled_width': width,
                    'scaled_height': height
                }),
                'objectid': ConstProperty(objectid)
            })

        for image in dataset.images:
            if max(image.width.value, image.height.value) <= self.max_edge:
                continue
            image._downscale_data = CachedProperty('downscale', "v1", self.method, self.max_edge, image)
            if image._downscale_data.has_value:
                mapping[image] = newimage(image)
            else:
                toconvert.append(image)

        if len(toconvert) == 0:
            self.log.info(f"No image need downscaling, skipping")
            dataset.images = [mapping.get(image, image) for image in dataset.images]
            return dataset

        for image in tqdm(toconvert, desc="Downscaling images"):
            if self.method == DownscaleMethod.PIL:
                image._downscale_data.value = self.downscale_image_pil(image, self.max_edge)
            else:
                raise ValueError(f"Unsupported downscaling method: {self.method}")

            mapping[image] = newimage(image)

        dataset.images = [mapping.get(image, image) for image in dataset.images]
        return dataset

__all__ = [ 'Downscale' ]

