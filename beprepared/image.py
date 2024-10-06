from beprepared.properties import PropertyBag, Property, ConstProperty
from typing import Dict

class Image(PropertyBag):
    ALLOWED_FORMATS = {'JPEG', 'PNG', 'WEBP', 'GIF', 'TIFF', 'BMP' }
