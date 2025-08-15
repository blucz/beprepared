from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.workspace import Abort
from beprepared.properties import CachedProperty, ConstProperty
from beprepared.nodes.utils import tqdm
from PIL import Image
from io import BytesIO
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

class DownscaleMethod:
    PIL    = "PIL"

class Downscale(Node):
    '''Downscales images to a specified edge length constraint.
    
    Can work with either max_edge (scales down if larger) or min_edge (scales down to exact minimum edge size).
    These parameters are mutually exclusive.
    '''

    def __init__(self, method=DownscaleMethod.PIL, max_edge=None, min_edge=None, format='PNG'):
        '''Initializes the Downscale node

        Args:
            method (str): The method to use for downscaling (e.g., 'PIL')
            max_edge (int, optional): The maximum edge length - scales down only if image is larger
            min_edge (int, optional): The minimum edge length - scales to make smallest edge exactly this size
            format (str): The format to save the downscaled images in (e.g., 'PNG', 'JPEG')
            
        Note:
            max_edge and min_edge are mutually exclusive. You must specify exactly one.
        '''
        super().__init__()
        
        if (max_edge is None) == (min_edge is None):
            raise ValueError("Exactly one of max_edge or min_edge must be specified")
            
        self.max_edge = max_edge
        self.min_edge = min_edge
        self.method   = method 
        self.format   = format

    def _downscale_worker(args):
        image_path, max_edge, min_edge, format = args
        try:
            # Load the image using Pillow
            image = Image.open(image_path)
            
            width, height = image.size
            
            if max_edge is not None:
                # Original behavior: scale down so largest edge is max_edge
                if max(width, height) <= max_edge:
                    # Image already small enough, return original
                    return {
                        'success': True,
                        'path': image_path,
                        'width': width,
                        'height': height,
                        'bytes': None,  # Signal no change needed
                        'skipped': True
                    }
                    
                # Calculate new dimensions
                if width > height:
                    new_width = max_edge
                    new_height = int((max_edge / width) * height)
                else:
                    new_height = max_edge
                    new_width = int((max_edge / height) * width)
                    
            else:  # min_edge is not None
                # New behavior: scale so smallest edge is exactly min_edge
                min_dimension = min(width, height)
                
                if min_dimension <= min_edge:
                    # Image already at or below target, return original
                    return {
                        'success': True,
                        'path': image_path,
                        'width': width,
                        'height': height,
                        'bytes': None,  # Signal no change needed
                        'skipped': True
                    }
                
                # Calculate scale factor based on minimum edge
                scale_factor = min_edge / min_dimension
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

            resized_image = image.resize((new_width, new_height), Image.LANCZOS)

            if resized_image.mode != 'RGB':
                resized_image = resized_image.convert('RGB')

            byte_array = BytesIO()
            resized_image.save(byte_array, format=format)
            
            return {
                'success': True,
                'path': image_path,
                'width': new_width,
                'height': new_height,
                'bytes': byte_array.getvalue(),
                'skipped': False
            }
        except Exception as e:
            return {
                'success': False,
                'path': image_path,
                'error': str(e),
                'skipped': False
            }

    def downscale_image_pil(self, image, edge_value, is_max_edge):
        image_path = self.workspace.get_path(image)
        result = self._downscale_worker((image_path, edge_value if is_max_edge else None, 
                                        edge_value if not is_max_edge else None, self.format))
        if result['bytes'] is None:
            # No change needed
            return None
        objectid = self.workspace.put_object(result['bytes'])
        return {
            'width': result['width'],
            'height': result['height'],
            'objectid': objectid
        }

    def eval(self, dataset) -> Dataset:
        toconvert = []
        mapping = { x: x for x in dataset.images }
        
        # Determine which edge constraint we're using
        edge_value = self.max_edge if self.max_edge is not None else self.min_edge
        is_max_edge = self.max_edge is not None
        edge_type = "max_edge" if is_max_edge else "min_edge"

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
                    edge_type: edge_value,
                    'original_width': image.width.value,
                    'original_height': image.height.value,
                    'scaled_width': width,
                    'scaled_height': height
                }),
                'objectid': ConstProperty(objectid)
            })

        for image in dataset.images:
            # Check if image needs processing
            if is_max_edge:
                # Skip if already small enough
                if max(image.width.value, image.height.value) <= edge_value:
                    continue
            else:
                # Skip if minimum edge is already at or below target
                if min(image.width.value, image.height.value) <= edge_value:
                    continue
                    
            image._downscale_data = CachedProperty('downscale', "v2", self.method, edge_type, edge_value, image)
            if image._downscale_data.has_value:
                mapping[image] = newimage(image)
            else:
                toconvert.append(image)

        if len(toconvert) == 0:
            self.log.info(f"No images need downscaling, skipping")
            dataset.images = [mapping.get(image, image) for image in dataset.images]
            return dataset

        # Process images in parallel using ProcessPoolExecutor
        num_workers = multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            if self.method == DownscaleMethod.PIL:
                futures = []
                for image in toconvert:
                    image_path = self.workspace.get_path(image)
                    futures.append(executor.submit(Downscale._downscale_worker, 
                                                 (image_path, 
                                                  edge_value if is_max_edge else None,
                                                  edge_value if not is_max_edge else None, 
                                                  self.format)))
                
                failed_images = []
                for image, future in tqdm(zip(toconvert, as_completed(futures)), total=len(toconvert), desc=f"Downscaling images ({num_workers} workers)"):
                    result = future.result()
                    if result['success']:
                        if result.get('skipped', False):
                            # Image didn't need scaling, keep original
                            continue
                        objectid = self.workspace.put_object(result['bytes'])
                        image._downscale_data.value = {
                            'width': result['width'],
                            'height': result['height'],
                            'objectid': objectid
                        }
                        mapping[image] = newimage(image)
                    else:
                        self.log.error(f"Error processing {self.workspace.get_path(image)}: {result['error']}")
                        failed_images.append(image)
                        del mapping[image]  # Remove failed image from mapping
            else:
                raise Abort(f"Unsupported downscaling method in Downscale: {self.method}")

        dataset.images = [mapping.get(image, image) for image in dataset.images]
        return dataset

__all__ = [ 'Downscale' ]