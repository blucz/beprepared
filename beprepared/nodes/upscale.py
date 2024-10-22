from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.workspace import Abort
from beprepared.properties import CachedProperty, ConstProperty
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

import torchvision.transforms
torchvision.transforms.functional_tensor = torchvision.transforms.functional

# TODO: cooler algorithms like SwinIR, also get ESRGAN working
class UpscaleMethod:
    PIL    = "PIL"
    ESRGAN = "ESRGAN"

class Upscale(Node):
    '''Upscales images to a specified minimum edge length'''
    def __init__(self, method=UpscaleMethod.PIL, min_edge=1024, format='PNG'):
        '''Initializes the Upscale node

        Args:
            method (str): The method to use for upscaling (e.g., 'PIL', 'ESRGAN')
            min_edge (int): The minimum edge length for the upscaling
            format (str): The format to save the upscaled images in (e.g., 'PNG', 'JPEG')

        **NOTE: The ESRGAN method is currently broken due to a bug in basicsr.**
        '''
        super().__init__()
        self.min_edge = min_edge
        self.method   = method 
        self.format   = format

    def upscale_image_pil(self, image, min_edge):
        # Load the image using Pillow
        image_path = self.workspace.get_path(image)
        image = Image.open(image_path)
        
        # Resize so that the shorter side is min_edge
        width, height = image.size
        if width < height:
            new_width = min_edge
            new_height = int((min_edge / width) * height)
        else:
            new_height = min_edge
            new_width = int((min_edge / height) * width)

        self.log.debug(f"Upscaling {image_path} using PIL.")
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

    def upscale_image_esrgan(self, image, min_edge):
        # Load the image using Pillow
        image_path = self.workspace.get_path(image)
        image = Image.open(image_path)
        
        width,height = image.size
        self.log.debug(f"Upscaling {image_path} using RealESRGAN.")
        self.log.debug(f"Original size: {width}x{height}")

        # Upscale the image
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        upscaled_image,_ = self.upsampler.enhance(cv2_image, outscale=2)
        resized_image = Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
        new_width, new_height = resized_image.size
        self.log.debug(f"Upscaled size: {new_width}x{new_height}")
        
        r_width, r_height = image.size
        if min(r_width, r_height) != min_edge:
            # Resize so that the shorter side is min_edge
            width, height = resized_image.size
            if width < height:
                new_width = min_edge
                new_height = int((min_edge / width) * height)
            else:
                new_height = min_edge
                new_width = int((min_edge / height) * width)

            resized_image = resized_image.resize((new_width, new_height), Image.LANCZOS)
            self.log.debug(f"Secondary scale to: {new_width}x{new_height}")
        else:
            self.log.debug(f"Skipping seocndary scale for {image_path} as the shortest side is already min_edge.")

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
            data = image._upscale_data.value
            width = data['width']
            height = data['height']
            objectid = data['objectid']
            return image.with_props({
                'width':  ConstProperty(width),
                'height': ConstProperty(height),
                'format': ConstProperty(self.format),
                'upscale_info': ConstProperty({
                    'method': self.method,
                    'min_edge': self.min_edge,
                    'original_width': image.width.value,
                    'original_height': image.height.value,
                    'scaled_width': width,
                    'scaled_height': height
                }),
                'objectid': ConstProperty(objectid)
            })

        for image in dataset.images:
            self.log.info(f"Checking {image.original_path.value}")
            if min(image.width.value, image.height.value) >= self.min_edge:
                self.log.info(f"Skipping upscaling for {image.original_path.value} as the shortest side is >= {self.min_edge}px.")
                continue
            image._upscale_data = CachedProperty('upscale', "v1", self.method, self.min_edge, image)
            if image._upscale_data.has_value:
                self.log.info(f"Using cached upscale data for {image.original_path.value}")
                self.log.info(f"data: {image._upscale_data.value}")
                mapping[image] = newimage(image)
            else:
                self.log.info(f"Need to upscale {image.original_path.value}")
                toconvert.append(image)

        if len(toconvert) == 0:
            self.log.info(f"No image need upscaling, skipping")
            dataset.images = [mapping.get(image, image) for image in dataset.images]
            return dataset

        if self.method == UpscaleMethod.ESRGAN:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            self.upsampler = RealESRGANer(model=model, model_path='RealESRGAN_x2plus.pth', scale=2)

        for image in tqdm(toconvert, desc="Upscaling images"):
            if self.method == UpscaleMethod.PIL:
                image._upscale_data.value = self.upscale_image_pil(image, self.min_edge)
            elif self.method == UpscaleMethod.ESRGAN:
                image._upscale_data.value = self.upscale_image_esrgan(image, self.min_edge)
            else:
                raise Abort(f"Unsupported upscaling method: {self.method}")
            mapping[image] = newimage(image)

        dataset.images = [mapping.get(image, image) for image in dataset.images]
        return dataset

__all__ = [ 'Upscale' ]

