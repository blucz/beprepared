import gc
from enum import Enum
from typing import Optional

from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from tqdm import tqdm
from PIL import Image
import json

from beprepared.node import Node 
from beprepared.properties import CachedProperty, ComputedProperty

class Florence2Task(Enum):
    """Available tasks for Florence-2 model"""
    CAPTION = "CAPTION"  # Basic object detection caption
    DETAILED_CAPTION = "DETAILED_CAPTION"  # Detailed caption
    MORE_DETAILED_CAPTION = "MORE_DETAILED_CAPTION"  # More detailed caption

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

        self.log.info(f"Generating captions for {len(needs_caption)} images")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        MODEL_NAME = "microsoft/Florence-2-large-ft"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        model = model.to(device)
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        task_token = f"<{self.task.value}>"

        for i in tqdm(range(0, len(needs_caption), self.batch_size), desc="Florence-2"):
            batch_images = needs_caption[i:i + self.batch_size]
            
            for image in batch_images:
                path = self.workspace.get_path(image)
                pil_image = Image.open(path).convert('RGB')

                inputs = processor(
                    text=task_token,
                    images=pil_image,
                    return_tensors="pt"
                ).to(device, torch_dtype)

                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=3,
                )
                
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

                parsed_caption = processor.post_process_generation(
                    generated_text,
                    task=self.task.value,
                    image_size=(pil_image.width, pil_image.height)
                )

                final_caption = parsed_caption[self.task.value]

                image._florence2_caption.value = final_caption
                self.log.info(f"{path} => {final_caption}")

        # Cleanup
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        
        return dataset

__all__ = ['Florence2Caption']
