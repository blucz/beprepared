import gc
import torch
import json
from io import BytesIO
from typing import List, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from beprepared.node import Node
from beprepared.properties import CachedProperty, ConstProperty
from beprepared.nodes.utils import tqdm
from .parallelworker import ParallelController, BaseWorker

@dataclass
class BatchItem:
    """Represents a batch of images to process"""
    image_indices: List[int]  # Original indices in the dataset
    image_paths: List[str]    # Paths to the images

class EdgeWatermarkRemovalWorker(BaseWorker):
    def initialize_worker(self):
        """Initialize the Florence-2 model and processor"""
        gpu_id = self.worker_params['gpu_id']
        torch.cuda.set_device(gpu_id)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large-ft",
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(f'cuda:{gpu_id}')
        
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large-ft",
            trust_remote_code=True
        )
        
        self.max_crop_percent = self.worker_params['max_crop_percent']
        self.preview_crop = self.worker_params['preview_crop']

    def _find_crop_region(self, image: Image.Image, bboxes: list) -> tuple[tuple[int, int, int, int], bool]:
        """Find a crop region that excludes watermarks completely within edge margins"""
        width, height = image.size
        edge_margin_x = int(width * self.max_crop_percent)
        edge_margin_y = int(height * self.max_crop_percent)
        
        # Filter bboxes to only those completely within edge margins
        edge_watermarks = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Define strict edge margin (5%) for initial filtering
            strict_margin_x = int(width * 0.05)
            strict_margin_y = int(height * 0.05)
            
            # Check if bbox comes within strict margin of any edge AND is completely within crop margin
            if ((x1 <= strict_margin_x or x2 >= width - strict_margin_x or 
                 y1 <= strict_margin_y or y2 >= height - strict_margin_y) and
                ((x1 <= edge_margin_x and x2 <= edge_margin_x) or
                 (x1 >= width - edge_margin_x and x2 >= width - edge_margin_x) or
                 (y1 <= edge_margin_y and y2 <= edge_margin_y) or
                 (y1 >= height - edge_margin_y and y2 >= height - edge_margin_y))):
                edge_watermarks.append((x1, y1, x2, y2))
        
        if not edge_watermarks:
            return (0, 0, width, height), False
            
        # For each edge, calculate minimum crop needed
        crops = []
        for x1, y1, x2, y2 in edge_watermarks:
            if x1 <= edge_margin_x:  # Left edge
                crops.append(('left', x2))
            if x2 >= width - edge_margin_x:  # Right edge
                crops.append(('right', width - x1))
            if y1 <= edge_margin_y:  # Top edge
                crops.append(('top', y2))
            if y2 >= height - edge_margin_y:  # Bottom edge
                crops.append(('bottom', height - y1))
        
        if not crops:
            return (0, 0, width, height), False
            
        # Group crops by edge
        edge_crops = {
            'left': [],
            'right': [],
            'top': [],
            'bottom': []
        }
        
        for edge, size in crops:
            edge_crops[edge].append(size)
            
        # For each edge that needs cropping, find minimum size needed
        # to remove all watermarks on that edge
        edge_sizes = {}
        if edge_crops['left']: edge_sizes['left'] = max(edge_crops['left'])
        if edge_crops['right']: edge_sizes['right'] = max(edge_crops['right'])
        if edge_crops['top']: edge_sizes['top'] = max(edge_crops['top'])
        if edge_crops['bottom']: edge_sizes['bottom'] = max(edge_crops['bottom'])
        
        if not edge_sizes:
            return (0, 0, width, height), False
            
        # Find edge requiring smallest crop
        best_edge, best_size = min(edge_sizes.items(), key=lambda x: x[1])
        
        # Apply crop on the edge that requires smallest removal
        new_left = max(edge_crops['left']) if best_edge == 'left' else 0
        new_right = width - max(edge_crops['right']) if best_edge == 'right' else width
        new_top = max(edge_crops['top']) if best_edge == 'top' else 0
        new_bottom = height - max(edge_crops['bottom']) if best_edge == 'bottom' else height
        
        return (new_left, new_top, new_right, new_bottom), True

    def _draw_bounding_boxes(self, image: Image.Image, bboxes: list, labels: list, crop_region: tuple[int, int, int, int] = None) -> tuple[Image.Image, bool]:
        """Draw bounding boxes around detected watermarks/text and crop region"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        did_draw = False
        
        # Draw detected watermark boxes in red
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            did_draw = True
        
        # Draw crop region in green if provided
        if crop_region:
            x1, y1, x2, y2 = crop_region
            # Draw the crop region with a thicker line and semi-transparent fill
            draw.rectangle([x1, y1, x2, y2], outline='green', width=5)
            # Draw a second rectangle slightly offset for better visibility
            draw.rectangle([x1+2, y1+2, x2-2, y2-2], outline='lightgreen', width=2)
            
        return image, did_draw

    def process_item(self, item: BatchItem) -> Tuple[List[int], List[dict]]:
        """Process a batch of images and return their results"""
        results = []
        
        for idx, path in zip(item.image_indices, item.image_paths):
            pil_image = Image.open(path).convert('RGB')

            # Use caption-to-phrase-grounding to find watermarks
            prompt = "<CAPTION_TO_PHRASE_GROUNDING>watermark or logo or text at the edge of the image"
            inputs = self.processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt"
            ).to(f'cuda:{self.worker_params["gpu_id"]}', torch.float16)

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_result = self.processor.post_process_generation(
                generated_text,
                task="<CAPTION_TO_PHRASE_GROUNDING>",
                image_size=(pil_image.width, pil_image.height)
            )

            # Process detected regions
            processed = pil_image.copy()
            did_modify = False
            result_data = {'did_modify': False}
            
            if parsed_result and "<CAPTION_TO_PHRASE_GROUNDING>" in parsed_result:
                result = parsed_result["<CAPTION_TO_PHRASE_GROUNDING>"]
                bboxes = result["bboxes"]
                labels = result["labels"]
                if bboxes:
                    crop_region, would_crop = self._find_crop_region(processed, bboxes)
                    
                    if self.preview_crop:
                        # Draw boxes and proposed crop region
                        processed, did_modify = self._draw_bounding_boxes(processed, bboxes, labels, crop_region if would_crop else None)
                    elif would_crop:
                        # Actually perform the crop
                        processed = processed.crop(crop_region)
                        did_modify = True

            if did_modify:
                # Save processed image
                output = BytesIO()
                processed.save(output, format='PNG')
                result_data = {
                    'objectid': output.getvalue(),  # Pass bytes directly
                    'did_modify': True,
                    'preview_mode': self.preview_crop,
                    'width': processed.width,
                    'height': processed.height,
                    'format': 'PNG',
                    'bboxes': parsed_result["<CAPTION_TO_PHRASE_GROUNDING>"]["bboxes"],
                    'labels': parsed_result["<CAPTION_TO_PHRASE_GROUNDING>"]["labels"]
                }

            results.append(result_data)

        return item.image_indices, results

    def shutdown_worker(self):
        """Clean up GPU resources"""
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()


class EdgeWatermarkRemoval(Node):
    '''Removes watermarks from images by detecting them with Florence-2 and cropping them out'''

    def __init__(self, max_crop_percent: float = 0.15, preview_crop: bool = False):
        '''Initialize the watermark removal node
        
        Args:
            max_crop_percent (float): Maximum percentage of width/height that can be cropped (0.0-1.0)
            preview_crop (bool): If True, draws bounding boxes and proposed crop region instead of cropping
        '''
        super().__init__()
        self.max_crop_percent = max_crop_percent
        self.preview_crop = preview_crop

    def _find_crop_region(self, image: Image.Image, bboxes: list) -> tuple[tuple[int, int, int, int], bool]:
        """Find a crop region that excludes watermarks completely within edge margins"""
        width, height = image.size
        edge_margin_x = int(width * self.max_crop_percent)
        edge_margin_y = int(height * self.max_crop_percent)
        
        # Filter bboxes to only those completely within edge margins
        edge_watermarks = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Define strict edge margin (5%) for initial filtering
            strict_margin_x = int(width * 0.05)
            strict_margin_y = int(height * 0.05)
            
            # Check if bbox comes within strict margin of any edge AND is completely within crop margin
            if ((x1 <= strict_margin_x or x2 >= width - strict_margin_x or 
                 y1 <= strict_margin_y or y2 >= height - strict_margin_y) and
                ((x1 <= edge_margin_x and x2 <= edge_margin_x) or
                 (x1 >= width - edge_margin_x and x2 >= width - edge_margin_x) or
                 (y1 <= edge_margin_y and y2 <= edge_margin_y) or
                 (y1 >= height - edge_margin_y and y2 >= height - edge_margin_y))):
                edge_watermarks.append((x1, y1, x2, y2))
                #self.log.info(f"Found watermark within edge margin: {x1},{y1},{x2},{y2}")
        
        if not edge_watermarks:
            return (0, 0, width, height), False
            
        # For each edge, calculate minimum crop needed
        crops = []
        for x1, y1, x2, y2 in edge_watermarks:
            if x1 <= edge_margin_x:  # Left edge
                crops.append(('left', x2))
            if x2 >= width - edge_margin_x:  # Right edge
                crops.append(('right', width - x1))
            if y1 <= edge_margin_y:  # Top edge
                crops.append(('top', y2))
            if y2 >= height - edge_margin_y:  # Bottom edge
                crops.append(('bottom', height - y1))
        
        if not crops:
            return (0, 0, width, height), False
            
        # Group crops by edge
        edge_crops = {
            'left': [],
            'right': [],
            'top': [],
            'bottom': []
        }
        
        for edge, size in crops:
            edge_crops[edge].append(size)
            
        # For each edge that needs cropping, find minimum size needed
        # to remove all watermarks on that edge
        edge_sizes = {}
        if edge_crops['left']: edge_sizes['left'] = max(edge_crops['left'])
        if edge_crops['right']: edge_sizes['right'] = max(edge_crops['right'])
        if edge_crops['top']: edge_sizes['top'] = max(edge_crops['top'])
        if edge_crops['bottom']: edge_sizes['bottom'] = max(edge_crops['bottom'])
        
        if not edge_sizes:
            return (0, 0, width, height), False
            
        # Find edge requiring smallest crop
        best_edge, best_size = min(edge_sizes.items(), key=lambda x: x[1])
        
        # Apply crop on the edge that requires smallest removal
        new_left = max(edge_crops['left']) if best_edge == 'left' else 0
        new_right = width - max(edge_crops['right']) if best_edge == 'right' else width
        new_top = max(edge_crops['top']) if best_edge == 'top' else 0
        new_bottom = height - max(edge_crops['bottom']) if best_edge == 'bottom' else height
        
        #self.log.info(f"Selected {best_edge} edge crop removing {best_size} pixels to clear all watermarks")
        return (new_left, new_top, new_right, new_bottom), True

    def _draw_bounding_boxes(self, image: Image.Image, bboxes: list, labels: list, crop_region: tuple[int, int, int, int] = None) -> tuple[Image.Image, bool]:
        """Draw bounding boxes around detected watermarks/text and crop region"""
        draw = ImageDraw.Draw(image)
        width, height = image.size
        did_draw = False
        
        # Draw detected watermark boxes in red
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            #self.log.info(f"Drew bbox for {label} at {x1},{y1},{x2},{y2}")
            did_draw = True
        
        # Draw crop region in green if provided
        if crop_region:
            x1, y1, x2, y2 = crop_region
            # Draw the crop region with a thicker line and semi-transparent fill
            draw.rectangle([x1, y1, x2, y2], outline='green', width=5)
            # Draw a second rectangle slightly offset for better visibility
            draw.rectangle([x1+2, y1+2, x2-2, y2-2], outline='lightgreen', width=2)
            #self.log.info(f"Drew crop region at {x1},{y1},{x2},{y2}")
            
        return image, did_draw

    def eval(self, dataset):
        # First pass - mark all images that need processing and prepare mapping
        needs_processing = []
        mapping = {x: x for x in dataset.images}
        
        for image in dataset.images:
            image._watermark_removal = CachedProperty('watermark_removal', 'v2', image)
            if not image._watermark_removal.has_value:
                needs_processing.append(image)
            elif image._watermark_removal.value['did_modify']:
                # Handle cached results
                data = image._watermark_removal.value
                mapping[image] = image.with_props({
                    'objectid': ConstProperty(data['objectid']),
                    'width': ConstProperty(data['width']),
                    'height': ConstProperty(data['height']),
                    'format': ConstProperty(data['format']),
                    'watermark_info': ConstProperty({
                        'did_modify': True,
                        'preview_mode': self.preview_crop,
                        'original_width': image.width.value,
                        'original_height': image.height.value,
                        'scaled_width': data['width'],
                        'scaled_height': data['height']
                    })
                })

        if len(needs_processing) == 0:
            self.log.info("No images need watermark removal, skipping")
            dataset.images = [mapping.get(image, image) for image in dataset.images]
            return dataset

        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        self.log.info(f"Processing {len(needs_processing)} images for watermark removal using {num_gpus} GPUs")
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available")

        # Create worker parameters for each GPU
        worker_params_list = [
            {
                'gpu_id': i,
                'max_crop_percent': self.max_crop_percent,
                'preview_crop': self.preview_crop
            }
            for i in range(num_gpus)
        ]

        # Create batches of work items (process one at a time due to memory constraints)
        batches = []
        for i, image in enumerate(needs_processing):
            batch = BatchItem(
                image_indices=[i],
                image_paths=[self.workspace.get_path(image)]
            )
            batches.append(batch)

        # Create and run the parallel controller
        controller = ParallelController(EdgeWatermarkRemovalWorker, worker_params_list)
        
        with tqdm(total=len(needs_processing), desc="Removing watermarks") as pbar:
            for success, result in controller.run(batches):
                if not success:
                    raise RuntimeError(f"Worker failed: {result}")
                
                indices, results = result
                for idx, result_data in zip(indices, results):
                    if result_data['did_modify']:
                        # Store the processed image
                        objectid = self.workspace.put_object(result_data['objectid'])
                        result_data['objectid'] = objectid
                    
                    needs_processing[idx]._watermark_removal.value = result_data
                pbar.update(len(indices))

        # Update all images with new properties
        mapping = {x: x for x in dataset.images}
        
        def newimage(image):
            data = image._watermark_removal.value
            if not data['did_modify']:
                return image
                
            return image.with_props({
                'objectid': ConstProperty(data['objectid']),
                'width': ConstProperty(data['width']),
                'height': ConstProperty(data['height']),
                'format': ConstProperty(data['format']),
                'watermark_info': ConstProperty({
                    'did_modify': True,
                    'preview_mode': self.preview_crop,
                    'original_width': image.width.value,
                    'original_height': image.height.value,
                    'scaled_width': data['width'],
                    'scaled_height': data['height']
                })
            })

        for image in needs_processing:
            mapping[image] = newimage(image)

        dataset.images = [mapping.get(image, image) for image in dataset.images]
        return dataset

__all__ = ['EdgeWatermarkRemoval']
