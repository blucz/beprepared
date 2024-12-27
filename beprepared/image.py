from beprepared.properties import PropertyBag, Property, ConstProperty
from typing import Dict

class Image(PropertyBag):
    ALLOWED_FORMATS = {'JPEG', 'PNG', 'WEBP', 'GIF', 'TIFF', 'BMP' }

    @property
    def aspect_ratio(self) -> float:
        """Returns the aspect ratio (width/height) of the image"""
        return ConstProperty(self.width.value / self.height.value)
