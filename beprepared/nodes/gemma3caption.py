"""Gemma 3 12B Vision-Language Model captioning node."""

import gc
from typing import Optional, List, Tuple
from dataclasses import dataclass

try:
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
except ImportError:
    raise ImportError(
        "Gemma 3 support requires transformers >= 4.46.0. "
        "Please run: pip install --upgrade transformers"
    )

import torch
from .utils import tqdm
from PIL import Image

from beprepared.node import Node 
from beprepared.properties import CachedProperty, ComputedProperty
from .parallelworker import ParallelController, BaseWorker

@dataclass
class BatchItem:
    """Represents a batch of images to process"""
    image_indices: List[int]  # Original indices in the dataset
    image_paths: List[str]    # Paths to the images

class Gemma3CaptionWorker(BaseWorker):
    def initialize_worker(self):
        """Initialize the Gemma 3 model and processor"""
        gpu_id = self.worker_params['gpu_id']
        torch.cuda.set_device(gpu_id)
        
        MODEL = "google/gemma-3-12b-it"
        
        # Load model with appropriate configuration
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL, 
            torch_dtype=torch.bfloat16,
            device_map=f'cuda:{gpu_id}',
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Load processor for handling both text and images
        self.processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        self.prompt = self.worker_params['prompt']

    def process_item(self, item: BatchItem) -> Tuple[List[int], List[str]]:
        """Process a batch of images and return their captions"""
        captions = []
        
        for path in item.image_paths:
            try:
                # Load and prepare image
                pil_image = Image.open(path).convert('RGB')
                
                # Prepare messages with image and text
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": self.prompt}
                        ]
                    }
                ]
                
                # Process the input using chat template
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                
                # Move to GPU and set dtype
                device = f'cuda:{self.worker_params["gpu_id"]}'
                inputs = {k: v.to(device, dtype=torch.bfloat16 if torch.is_floating_point(v) else v.dtype) 
                         for k, v in inputs.items()}
                
                # Get the input length for trimming later
                input_len = inputs["input_ids"].shape[-1]
                
                # Generate response
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.5,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1
                    )
                    
                    # Extract only the generated tokens (exclude input)
                    generated_tokens = generated_ids[0][input_len:]
                    
                    # Decode the response
                    caption = self.processor.decode(generated_tokens, skip_special_tokens=True)
                    captions.append(caption.strip())
                    
            except Exception as e:
                # If there's an error processing an image, add an error message
                captions.append(f"Error processing image: {str(e)}")
                print(f"Error processing {path}: {e}")
        
        return item.image_indices, captions

    def shutdown_worker(self):
        """Clean up GPU resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        gc.collect()
        torch.cuda.empty_cache()


class Gemma3Caption(Node):
    '''Generates image captions using Google's Gemma 3 12B Instruction-Tuned Vision-Language Model.
    
    Gemma 3 is a multimodal model that can process both text and images, generating
    high-quality text outputs. It supports a 128K token context window and is
    multilingual, supporting over 140 languages.
    
    The model excels at:
    - Detailed image descriptions
    - Visual question answering
    - Image analysis and reasoning
    - Content understanding
    
    Note: Requires transformers >= 4.46.0
    '''
    
    DEFAULT_PROMPT = "Describe this image in detail, including the subject, composition, colors, lighting, and any notable features or context."

    def __init__(self,
                 target_prop: str = 'caption',
                 prompt: Optional[str] = None,
                 instructions: Optional[str] = None,
                 batch_size: int = 1):
        '''Initializes the Gemma3Caption node

        Args:
            target_prop (str): The property to store the caption in (default: 'caption')
            prompt (str): The prompt to use for the Gemma 3 model 
                         (default: detailed description prompt)
            instructions (str): Additional instructions to append to the prompt (optional)
            batch_size (int): The number of images to process in parallel. 
                             Note: Gemma 3 12B requires significant VRAM, so batch_size=1 is recommended
        '''
        super().__init__()
        self.target_prop = target_prop
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.batch_size = batch_size
        if instructions:
            self.prompt = f"{self.prompt}\n\n{instructions}"

    def eval(self, dataset):
        needs_caption = []
        for image in dataset.images:
            # Create cached property with system prompt included in cache key
            image._gemma3_caption = CachedProperty('gemma3-12b-it', 'v1', self.prompt, image)
            # Use lambda with image parameter like other caption nodes
            setattr(image, self.target_prop, ComputedProperty(lambda image: image._gemma3_caption.value if image._gemma3_caption.has_value else None))
            if not image._gemma3_caption.has_value:
                needs_caption.append(image)

        if len(needs_caption) == 0:
            self.log.info("All images already have captions, skipping")
            return dataset

        self.log.info(f"Generating captions for {len(needs_caption)} images using Gemma 3 12B")

        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available. Gemma 3 12B requires a GPU with sufficient VRAM (24GB+ recommended).")

        # Create worker parameters for each GPU
        worker_params_list = [
            {
                'gpu_id': i, 
                'prompt': self.prompt,
            }
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
        controller = ParallelController(Gemma3CaptionWorker, worker_params_list)
        
        with tqdm(total=len(needs_caption), desc="Gemma 3 Vision") as pbar:
            for success, result in controller.run(batches):
                if not success:
                    raise RuntimeError(f"Worker failed: {result}")
                
                indices, captions = result
                for idx, caption in zip(indices, captions):
                    needs_caption[idx]._gemma3_caption.value = caption
                    self.log.info(f"{self.workspace.get_path(needs_caption[idx])} => {caption}")
                pbar.update(len(indices))
        
        return dataset

__all__ = ['Gemma3Caption']
