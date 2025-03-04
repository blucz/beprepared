import os
import gc
from typing import Optional, List, Tuple
from dataclasses import dataclass
import multiprocessing
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
import torch

from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.workspace import Abort
from beprepared.properties import CachedProperty, ConstProperty
from beprepared.nodes.utils import tqdm
from .parallelworker import ParallelController, BaseWorker
from .centerface import CenterFace

@dataclass
class BatchItem:
    """Represents a batch of images to process"""
    image_indices: List[int]  # Original indices in the dataset
    image_paths: List[str]    # Paths to the images


class AnonymizeMethod:
    BLUR = "blur"
    SOLID = "solid"
    NONE = "none"
    MOSAIC = "mosaic"


class AnonymizeWorker(BaseWorker):
    def initialize_worker(self):
        """Initialize the worker with GPU settings"""
        gpu_id = self.worker_params['gpu_id']
        torch.cuda.set_device(gpu_id)
        
        self.method = self.worker_params['method']
        self.threshold = self.worker_params['threshold']
        self.mask_scale = self.worker_params['mask_scale']
        self.ellipse = self.worker_params['ellipse']
        self.mosaic_size = self.worker_params['mosaic_size']
        
        # Initialize CenterFace on GPU if possible
        self.centerface = CenterFace(in_shape=None, backend='auto')
        
        # Import skimage for elliptical masking if needed
        if self.ellipse:
            import skimage.draw
            self.skimage = skimage
    
    def process_item(self, item: Tuple[int,str]) -> Tuple[List[int], List[bytes]]:
        """Process a batch of images and return their anonymized versions"""
        idx,path = item
        
        # Load the image using OpenCV
        frame = cv2.imread(path)
        if frame is None:
            self.log.error(f"Failed to load image: {path}")
            return -1,None,None
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        dets, _ = self.centerface(frame, threshold=self.threshold)
        
        # Anonymize faces
        for i, det in enumerate(dets):
            boxes, score = det[:4], det[4]
            x1, y1, x2, y2 = boxes.astype(int)
            
            # Scale bounding box
            h, w = y2 - y1, x2 - x1
            y1 -= int(h * (self.mask_scale - 1.0) / 2)
            y2 += int(h * (self.mask_scale - 1.0) / 2)
            x1 -= int(w * (self.mask_scale - 1.0) / 2)
            x2 += int(w * (self.mask_scale - 1.0) / 2)
            
            # Clip to image boundaries
            y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
            x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)

            # Apply anonymization
            if self.method == AnonymizeMethod.BLUR:
                bf = 2  # blur factor
                blurred_box = cv2.blur(
                    frame[y1:y2, x1:x2],
                    (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
                )
                if self.ellipse:
                    roibox = frame[y1:y2, x1:x2]
                    # Get y and x coordinate lists of the "bounding ellipse"
                    ey, ex = self.skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
                    # Filter coordinates to be within bounds
                    valid_indices = (ey < roibox.shape[0]) & (ex < roibox.shape[1])
                    ey, ex = ey[valid_indices], ex[valid_indices]
                    roibox[ey, ex] = blurred_box[ey, ex]
                    frame[y1:y2, x1:x2] = roibox
                else:
                    frame[y1:y2, x1:x2] = blurred_box
            elif self.method == AnonymizeMethod.SOLID:
                print("    solid")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
            elif self.method == AnonymizeMethod.MOSAIC:
                print("    mosaic")
                for y in range(y1, y2, self.mosaic_size):
                    for x in range(x1, x2, self.mosaic_size):
                        pt1 = (x, y)
                        pt2 = (min(x2, x + self.mosaic_size - 1), min(y2, y + self.mosaic_size - 1))
                        color = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                        cv2.rectangle(frame, pt1, pt2, color, -1)
            elif self.method == AnonymizeMethod.NONE:
                print("    none")
                pass
        
        # Convert to PIL Image and save to bytes
        pil_img = Image.fromarray(frame)
        byte_array = BytesIO()
        pil_img.save(byte_array, format='PNG')
        print("save")
        
        return idx, dets, byte_array.getvalue()
    
    def shutdown_worker(self):
        """Clean up GPU resources"""
        del self.centerface
        gc.collect()
        torch.cuda.empty_cache()


class Anonymize(Node):
    '''Anonymizes faces in images using the CenterFace face detection model'''

    def __init__(self, 
                 method=AnonymizeMethod.BLUR, 
                 threshold=0.4, 
                 mask_scale=1.0, 
                 ellipse=True, 
                 mosaic_size=20):
        '''Initializes the Anonymize node

        Args:
            method (str): The method to use for anonymization ('blur', 'solid', 'none', 'mosaic')
            threshold (float): Detection threshold for face detection (0.0-1.0)
            mask_scale (float): Scale factor for face masks to ensure complete face coverage
            ellipse (bool): Use elliptical masks instead of rectangular ones
            mosaic_size (int): Size of mosaic blocks when using mosaic method
        '''
        super().__init__()
        self.method = method
        self.threshold = threshold
        self.mask_scale = mask_scale
        self.ellipse = ellipse
        self.mosaic_size = mosaic_size

    def eval(self, dataset) -> Dataset:
        toconvert = []
        mapping = {x: x for x in dataset.images}

        def newimage(image): 
            data = image._anonymize_data.value
            objectid = data['objectid']
            return image.with_props({
                'objectid': ConstProperty(objectid)
            })

        for image in dataset.images:
            image._anonymize_data = CachedProperty('anonymize', "v1", self.method, self.threshold, self.mask_scale, self.ellipse, self.mosaic_size, image)
            if image._anonymize_data.has_value:
                mapping[image] = newimage(image)
            else:
                toconvert.append(image)

        if len(toconvert) == 0:
            self.log.info(f"No images need anonymization, skipping")
            dataset.images = [mapping.get(image, image) for image in dataset.images]
            return dataset

        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            self.log.warning("No CUDA devices available, using CPU only")
            num_gpus = 1  # Fall back to CPU
        
        self.log.info(f"Processing {len(toconvert)} images with {num_gpus} {'GPU' if num_gpus == 1 else 'GPUs'}")
        
        # Create worker parameters for each GPU
        worker_params_list = [
            {
                'gpu_id': i % num_gpus,
                'method': self.method,
                'threshold': self.threshold,
                'mask_scale': self.mask_scale,
                'ellipse': self.ellipse,
                'mosaic_size': self.mosaic_size
            }
            for i in range(num_gpus)
        ]

        items = []
        for i,image in enumerate(toconvert):
            items.append((i, self.workspace.get_path(image)))

        # Create and run the parallel controller
        controller = ParallelController(AnonymizeWorker, worker_params_list)
        
        with tqdm(total=len(toconvert), desc="Anonymizing faces") as pbar:
            for success, result in controller.run(items):
                if not success:
                    raise RuntimeError(f"Worker failed: {result}")
                
                idx, dets, image_bytes = result
                if image_bytes is None:
                    self.log.error(f"Failed to process {self.workspace.get_path(toconvert[idx])}")
                    del mapping[toconvert[idx]]
                    continue
                    
                objectid = self.workspace.put_object(image_bytes)
                toconvert[idx]._anonymize_data.value = {
                    'objectid': objectid,
                    'face_bboxes': dets
                }
                mapping[toconvert[idx]] = newimage(toconvert[idx])
                pbar.update(1)
        
        dataset.images = [mapping.get(image, image) for image in dataset.images]
        return dataset

__all__ = ['Anonymize', 'AnonymizeMethod']
