import os
import gc
from typing import Optional, Literal, Dict, List, Tuple
from dataclasses import dataclass

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

import torch
from .utils import tqdm
from PIL import Image

from beprepared.workspace import Workspace
from beprepared.node import Node 
from beprepared.properties import CachedProperty, ComputedProperty
from beprepared.nodes.convert_format import convert_image
from qwen_vl_utils import process_vision_info
from .parallelworker import ParallelController, BaseWorker

QWENVL_IMAGE_SIZE = 768

@dataclass
class BatchItem:
    """Represents a batch of images to process"""
    image_indices: List[int]  # Original indices in the dataset
    image_paths: List[str]    # Paths to the images

class QwenVLCaptionWorker(BaseWorker):
    def initialize_worker(self):
        """Initialize the Qwen-VL model and processor"""
        gpu_id = self.worker_params['gpu_id']
        torch.cuda.set_device(gpu_id)
        
        MODEL = "Qwen/Qwen2-VL-7B-Instruct"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        )
        self.model.to(f'cuda:{gpu_id}')
        self.processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        self.prompt = self.worker_params['prompt']

    def process_item(self, item: BatchItem) -> Tuple[List[int], List[str]]:
        """Process a batch of images and return their captions"""
        pil_images = []
        for path in item.image_paths:
            img = Image.open(path).convert('RGB')
            width, height = img.size
            if width < height:
                new_width = QWENVL_IMAGE_SIZE
                new_height = int(height * (QWENVL_IMAGE_SIZE / width))
            else:
                new_height = QWENVL_IMAGE_SIZE
                new_width = int(width * (QWENVL_IMAGE_SIZE / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            pil_images.append(img)

        messages_batch = []
        text_batch = []
        for img in pil_images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        { "type": "image", "image": img, },
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            messages_batch.append(messages)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            text_batch.append(text)

        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = self.processor(
            text=text_batch,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(f'cuda:{self.worker_params["gpu_id"]}')

        generated_ids = self.model.generate(**inputs, max_new_tokens=300, temperature=0.5)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return item.image_indices, [caption.strip() for caption in output_text]

    def shutdown_worker(self):
        """Clean up GPU resources"""
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()


class QwenVLCaption(Node):
    '''Generates image captions using the Qwen 2 VL 7B model'''
    DEFAULT_PROMPT = "Write 4-6 sentences that describe this image. Include any relevant details about the composition, framing, quality, and subject of the image, as well as a full description of the scene. Only describe what you see in the image, and do not include any personal opinions or assumptions."

    def __init__(self,
                 target_prop:    str                              = 'caption',
                 prompt:         Optional[str] = None,
                 instructions:   Optional[str] = None,
                 batch_size:     int           = 1):
        '''Initializes the QwenVLCaption node

        Args:
            target_prop (str): The property to store the caption in (default is 'caption')
            prompt (str): The prompt to use for the Qwen 2 VL 7B model (read the code)
            instructions (str): Additional instructions to include in the prompt
            batch_size (int): The number of images to process in parallel. If you are running out of memory, try reducing this value.
        '''
        super().__init__()
        self.target_prop  = target_prop
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.batch_size = batch_size
        if instructions:
            self.prompt = f"{self.prompt}\n\n{instructions}"

    def eval(self, dataset):
        needs_caption = []
        for image in dataset.images:
            image._qwenvl_caption = CachedProperty('qwen-vl', 'v1', self.prompt, image)
            setattr(image, self.target_prop, ComputedProperty(lambda image: image._qwenvl_caption.value if image._qwenvl_caption.has_value else None))
            if not image._qwenvl_caption.has_value:
                needs_caption.append(image)

        if len(needs_caption) == 0: 
            self.log.info("All images already have captions, skipping")
            return dataset

        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        self.log.info(f"Processing {len(needs_caption)} images with Qwen-VL using {num_gpus} GPUs")
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available")

        # Create worker parameters for each GPU
        worker_params_list = [
            {'gpu_id': i, 'prompt': self.prompt}
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
        controller = ParallelController(QwenVLCaptionWorker, worker_params_list)
        
        with tqdm(total=len(needs_caption), desc="Qwen-VL") as pbar:
            for success, result in controller.run(batches):
                if not success:
                    raise RuntimeError(f"Worker failed: {result}")
                
                indices, captions = result
                for idx, caption in zip(indices, captions):
                    needs_caption[idx]._qwenvl_caption.value = caption
                    self.log.info(f"Generated caption for {needs_caption[idx].objectid.value}: {caption}")
                pbar.update(len(indices))
        
        return dataset

