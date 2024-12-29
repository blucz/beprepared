from io import BytesIO
from PIL import Image as PILImage
from beprepared import Image, Dataset, Node
from .utils import tqdm

from beprepared.properties import ConstProperty, CachedProperty

def convert_image(input_image_path, output_format):
    #print(f"Converting {input_image_path} to {output_format}")
    TRANSPARENCY_SUPPORTED_FORMATS = {'PNG', 'WEBP', 'GIF'}
    # Open the image
    image = PILImage.open(input_image_path)

    # Convert the output format to uppercase to match standard format names
    output_format = output_format.upper()

    # Check if the output format supports transparency
    transparency_supported = output_format in TRANSPARENCY_SUPPORTED_FORMATS

    # Handle RGBA and LA modes (with transparency) only if the output format does not support transparency
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        if not transparency_supported:
            # Convert RGBA/LA to RGB with a white background since the output format doesn't support transparency
            background = PILImage.new('RGBA', image.size, (255, 255, 255, 255))  # White background, fully opaque
            image = PILImage.alpha_composite(background, image.convert('RGBA'))
            image = image.convert('RGB')
        else:
            # No need to flatten transparency, keep it as RGBA
            image = image.convert('RGBA')
    else:
        # For all other cases, convert to RGB if necessary (e.g., grayscale, palette)
        if image.mode != 'RGB':
            image = image.convert('RGB')

    # Save the image to a byte array in the desired format
    byte_array = BytesIO()
    image.save(byte_array, format=output_format)

    # Get the byte data
    converted_data = byte_array.getvalue()

    return converted_data

class ConvertFormat(Node):
    '''Converts images to a specified format'''

    def __init__(self, format: str):
        '''Initializes the ConvertFormat node

        Args:
            format (str): The format to convert the images to (e.g., 'PNG', 'JPEG', 'WEBP')
        '''
        super().__init__()
        self.format = format

    def eval(self, dataset) -> Dataset:
        toconvert = []
        mapping = { x: x for x in dataset.images }
        for image in dataset.images:
            if image.format.value == self.format:
                continue
            image._convert_format_hash = CachedProperty('convertformat', "v1", self.format, image)
            if image._convert_format_hash.has_value:
                newimage = image.with_props({
                    'format': ConstProperty(self.format),
                    'objectid': image._convert_format_hash
                })
                mapping[image] = newimage
            else:
                toconvert.append(image)

        if len(toconvert) == 0:
            self.log.info(f"All images are already in the desired format, skipping")
            return dataset

        for image in tqdm(toconvert, desc="Converting images"):
            newbytes = convert_image(self.workspace.get_path(image), self.format)
            image._convert_format_hash.value = self.workspace.put_object(newbytes)
            newimage = image.with_props({
                'format': ConstProperty(self.format),
                'objectid': image._convert_format_hash
            })
            mapping[image] = newimage

        dataset.images = [mapping.get(image, image) for image in dataset.images]
        return dataset

__all__ = [ 'ConvertFormat' ]
