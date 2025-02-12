import gc
from typing import Optional, List, Tuple
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
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

class MolmoCaptionWorker(BaseWorker):
    def initialize_worker(self):
        """Initialize the Molmo model and processor"""
        gpu_id = self.worker_params['gpu_id']
        torch.cuda.set_device(gpu_id)
        
        MODEL = "allenai/Molmo-7B-D-0924"
        
        self.processor = AutoProcessor.from_pretrained(
            MODEL,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(f'cuda:{gpu_id}')
        
        self.prompt = self.worker_params['prompt']

    def process_item(self, item: BatchItem) -> Tuple[List[int], List[str]]:
        """Process a batch of images and return their captions"""
        pil_images = []
        for path in item.image_paths:
            img = Image.open(path).convert('RGB')
            pil_images.append(img)

        # Process inputs
        inputs = self.processor.process(
            images=pil_images,
            text=self.prompt
        )
        
        # Move inputs to device and batch
        inputs = {k: v.to(f'cuda:{self.worker_params["gpu_id"]}').unsqueeze(0) for k, v in inputs.items()}

        # Generate captions
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )

        # Decode captions
        captions = []
        for idx in range(len(pil_images)):
            generated_tokens = output[idx, inputs['input_ids'].size(1):]
            caption = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            captions.append(caption.strip())

        return item.image_indices, captions

    def shutdown_worker(self):
        """Clean up GPU resources"""
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()


class MolmoCaption(Node):
    '''Generates image captions using the Molmo-7B-D-0924 model'''
    DEFAULT_PROMPT = "Describe the contents and style of this image. Do not make inferences or draw conclusions, simply describe what is there without bias."

    def __init__(self,
                 target_prop: str = 'caption',
                 prompt: Optional[str] = None,
                 instructions: Optional[str] = None,
                 batch_size: int = 1):
        '''Initializes the MolmoCaption node

        Args:
            target_prop (str): The property to store the caption in
            prompt (str): The prompt to use for the Molmo model
            instructions (str): Additional instructions to include in the prompt
            batch_size (int): The number of images to process in parallel. If you are running out of memory, try reducing this value.
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
            image._molmo_caption = CachedProperty('molmo-7B-D-0924', 'v1', self.prompt, image)
            setattr(image, self.target_prop, ComputedProperty(lambda image: image._molmo_caption.value if image._molmo_caption.has_value else None))
            if not image._molmo_caption.has_value:
                needs_caption.append(image)

        if len(needs_caption) == 0:
            self.log.info("All images already have captions, skipping")
            return dataset

        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        self.log.info(f"Processing {len(needs_caption)} images with Molmo using {num_gpus} GPUs")
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
        controller = ParallelController(MolmoCaptionWorker, worker_params_list)
        
        with tqdm(total=len(needs_caption), desc="Molmo") as pbar:
            for success, result in controller.run(batches):
                if not success:
                    raise RuntimeError(f"Worker failed: {result}")
                
                indices, captions = result
                for idx, caption in zip(indices, captions):
                    needs_caption[idx]._molmo_caption.value = caption
                    self.log.info(f"{self.workspace.get_path(needs_caption[idx])} => {caption}")
                pbar.update(len(indices))
        
        return dataset

__all__ = ['MolmoCaption']
