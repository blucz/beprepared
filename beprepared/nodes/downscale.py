from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.workspace import Abort
from beprepared.properties import CachedProperty, ConstProperty
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

class DownscaleMethod:
    PIL    = "PIL"

class Downscale(Node):
    '''Downscales images to a specified maximum edge length'''

    def __init__(self, method=DownscaleMethod.PIL, max_edge=1024, format='PNG'):
        '''Initializes the Downscale node

        Args:
            method (str): The method to use for downscaling (e.g., 'PIL')
            max_edge (int): The maximum edge length for the downscaling
            format (str): The format to save the downscaled images in (e.g., 'PNG', 'JPEG')
        '''
        super().__init__()
        self.max_edge = max_edge
        self.method   = method 
        self.format   = format

    def _downscale_worker(args):
        image_path, max_edge, format = args
        # Load the image using Pillow
        image = Image.open(image_path)
        
        # Resize so that the shorter side is max_edge
        width, height = image.size
        if width > height:
            new_width = max_edge
            new_height = int((max_edge / width) * height)
        else:
            new_height = max_edge
            new_width = int((max_edge / height) * width)

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        if resized_image.mode != 'RGB':
            resized_image = resized_image.convert('RGB')

        byte_array = BytesIO()
        resized_image.save(byte_array, format=format)
        
        return {
            'path': image_path,
            'width': new_width,
            'height': new_height,
            'bytes': byte_array.getvalue()
        }

    def downscale_image_pil(self, image, max_edge):
        image_path = self.workspace.get_path(image)
        result = self._downscale_worker((image_path, max_edge, self.format))
        objectid = self.workspace.put_object(result['bytes'])
        return {
            'width': result['width'],
            'height': result['height'],
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

        # Process images in parallel using ProcessPoolExecutor
        num_workers = multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            if self.method == DownscaleMethod.PIL:
                futures = []
                for image in toconvert:
                    image_path = self.workspace.get_path(image)
                    futures.append(executor.submit(Downscale._downscale_worker, (image_path, self.max_edge, self.format)))
                
                for image, future in tqdm(zip(toconvert, as_completed(futures)), total=len(toconvert), desc=f"Downscaling images ({num_workers} workers)"):
                    try:
                        result = future.result()
                        objectid = self.workspace.put_object(result['bytes'])
                        image._downscale_data.value = {
                            'width': result['width'],
                            'height': result['height'],
                            'objectid': objectid
                        }
                        mapping[image] = newimage(image)
                    except Exception as e:
                        self.log.error(f"Error processing {self.workspace.get_path(image)}: {str(e)}")
                        raise
            else:
                raise Abort(f"Unsupported downscaling method in Downscale: {self.method}")

        dataset.images = [mapping.get(image, image) for image in dataset.images]
        return dataset

__all__ = [ 'Downscale' ]

