import gc
from enum import Enum
from typing import Optional, List, Tuple
from dataclasses import dataclass

from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from beprepared.nodes.utils import tqdm
from PIL import Image
import json

from beprepared.node import Node 
from beprepared.properties import CachedProperty, ComputedProperty
from .parallelworker import ParallelController, BaseWorker

@dataclass
class BatchItem:
    """Represents a batch of images to process"""
    image_indices: List[int]  # Original indices in the dataset
    image_paths: List[str]    # Paths to the images

class Florence2Task(Enum):
    """Available tasks for Florence-2 model"""
    CAPTION = "CAPTION"  # Basic object detection caption
    DETAILED_CAPTION = "DETAILED_CAPTION"  # Detailed caption
    MORE_DETAILED_CAPTION = "MORE_DETAILED_CAPTION"  # More detailed caption

class Florence2CaptionWorker(BaseWorker):
    def initialize_worker(self):
        """Initialize the Florence-2 model and processor"""
        gpu_id = self.worker_params['gpu_id']
        torch.cuda.set_device(gpu_id)
        
        MODEL_NAME = "microsoft/Florence-2-large-ft"
        torch_dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        self.model.to(f'cuda:{gpu_id}')
        
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        self.task = self.worker_params['task']
        self.task_token = f"<{self.task.value}>"

    def process_item(self, item: BatchItem) -> Tuple[List[int], List[str]]:
        """Process a batch of images and return their captions"""
        captions = []
        for path in item.image_paths:
            pil_image = Image.open(path).convert('RGB')
            
            inputs = self.processor(
                text=self.task_token,
                images=pil_image,
                return_tensors="pt"
            ).to(f'cuda:{self.worker_params["gpu_id"]}', torch.float16)

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            parsed_caption = self.processor.post_process_generation(
                generated_text,
                task=self.task.value,
                image_size=(pil_image.width, pil_image.height)
            )

            final_caption = parsed_caption[self.task.value]
            captions.append(final_caption)

        return item.image_indices, captions

    def shutdown_worker(self):
        """Clean up GPU resources"""
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()

class Florence2Caption(Node):
    '''Generates image captions using the Florence-2-base model'''

    def __init__(self,
                 target_prop: str = 'caption',
                 task: Florence2Task = Florence2Task.MORE_DETAILED_CAPTION,
                 batch_size: int = 64):
        '''Initializes the Florence2Caption node

        Args:
            target_prop (str): The property to store the caption in
            task (Florence2Task): The captioning task to perform (CAPTION, DETAILED_CAPTION, or MORE_DETAILED_CAPTION)
            batch_size (int): The number of images to process in parallel.
        '''
        super().__init__()
        self.target_prop = target_prop
        self.task = task
        self.batch_size = batch_size

    def eval(self, dataset):
        needs_caption = []
        for image in dataset.images:
            image._florence2_caption = CachedProperty('florence2-large', 'v1', self.task.value, image)
            setattr(image, self.target_prop, ComputedProperty(lambda image: image._florence2_caption.value if image._florence2_caption.has_value else None))
            if not image._florence2_caption.has_value:
                needs_caption.append(image)

        if len(needs_caption) == 0:
            self.log.info("All images already have captions, skipping")
            return dataset

        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        self.log.info(f"Processing {len(needs_caption)} images with Florence-2 using {num_gpus} GPUs")
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available")

        # Create worker parameters for each GPU
        worker_params_list = [
            {'gpu_id': i, 'task': self.task}
            for i in range(num_gpus)
        ]

        # Create batches of work items
        batches = []
        for i in range(0, len(needs_caption), self.batch_size):
            batch_images = needs_caption[i:i + self.batch_size]
            batch = BatchItem(
                image_indices=list(range(i, min(i + self.batch_size, len(needs_caption)))),
                image_paths=[self.workspace.get_path(img) for img in batch_images]
            )
            batches.append(batch)

        # Create and run the parallel controller
        controller = ParallelController(Florence2CaptionWorker, worker_params_list)
        
        with tqdm(total=len(needs_caption), desc="Florence-2") as pbar:
            for success, result in controller.run(batches):
                if not success:
                    raise RuntimeError(f"Worker failed: {result}")
                
                indices, captions = result
                for idx, caption in zip(indices, captions):
                    needs_caption[idx]._florence2_caption.value = caption
                    self.log.info(f"{self.workspace.get_path(needs_caption[idx])} => {caption}")
                pbar.update(len(indices))
        
        return dataset

__all__ = ['Florence2Caption']
