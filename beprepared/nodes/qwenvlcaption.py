import os
import gc
from typing import Optional, Literal

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

import torch
from .utils import tqdm
from PIL import Image

from beprepared.workspace import Workspace
from beprepared.node import Node 
from beprepared.properties import CachedProperty, ComputedProperty
from beprepared.nodes.convert_format import convert_image
from qwen_vl_utils import process_vision_info

QWENVL_IMAGE_SIZE = 768

class QwenVLCaption(Node):
    '''Generates image captions using the Qwen 2 VL 7B model'''
    DEFAULT_PROMPT = """Describe the contents and style of this image."""

    def __init__(self,
                 target_prop:    str           = 'caption',
                 prompt:         Optional[str]  = None,
                 instructions:   Optional[str]  = None,
                 batch_size:     int           = 1,
                 device_ids:     List[int]     = None):
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
        self.device_ids = device_ids or [0]  # Default to first GPU if none specified
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

        MODEL = "Qwen/Qwen2-VL-7B-Instruct"
        
        # Initialize models on each GPU
        models = []
        processors = []
        for device_id in self.device_ids:
            device = f'cuda:{device_id}'
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                    MODEL, 
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=False,
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
            models.append(model)
            processors.append(processor)

        # Calculate effective batch size across all GPUs
        total_batch_size = self.batch_size * len(self.device_ids)
        
        for i in tqdm(range(0, len(needs_caption), total_batch_size), desc="Qwen-VL"):
            batch_images = needs_caption[i:i + total_batch_size]
            pil_images = []
            for image in batch_images:
                path = self.workspace.get_path(image)
                img = Image.open(path).convert('RGB')
                # Resize so shorter side matches Qwen-VL expected size
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
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                text_batch.append(text)

            image_inputs, video_inputs = process_vision_info(messages_batch)
            inputs = processor(
                text=text_batch,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            # Split batch across GPUs
            sub_batches = []
            for j in range(len(self.device_ids)):
                start_idx = j * self.batch_size
                end_idx = start_idx + self.batch_size
                if start_idx < len(batch_images):
                    sub_batches.append(batch_images[start_idx:end_idx])
                    
            all_output_text = []
            
            # Process each sub-batch on its designated GPU
            for sub_batch, model, processor, device_id in zip(sub_batches, models, processors, self.device_ids):
                if not sub_batch:
                    continue
                    
                device = f'cuda:{device_id}'
                
                # Process sub-batch
                sub_messages_batch = []
                sub_text_batch = []
                for img in sub_batch:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                { "type": "image", "image": img, },
                                {"type": "text", "text": self.prompt},
                            ],
                        }
                    ]
                    sub_messages_batch.append(messages)
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    sub_text_batch.append(text)

                sub_image_inputs, sub_video_inputs = process_vision_info(sub_messages_batch)
                sub_inputs = processor(
                    text=sub_text_batch,
                    images=sub_image_inputs,
                    videos=sub_video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                sub_inputs = sub_inputs.to(device)

                sub_generated_ids = model.generate(**sub_inputs, max_new_tokens=300, temperature=0.5)
                sub_generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(sub_inputs.input_ids, sub_generated_ids)
                ]
                sub_output_text = processor.batch_decode(
                    sub_generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                all_output_text.extend(sub_output_text)

            output_text = all_output_text

            for image, caption in zip(batch_images, output_text):
                image._qwenvl_caption.value = caption.strip()

        # Cleanup all models and processors
        for model in models:
            del model
        del models
        del processors
        gc.collect()
        torch.cuda.empty_cache()
        
        return dataset

