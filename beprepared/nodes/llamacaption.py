import gc
from typing import Optional, List, Tuple
from dataclasses import dataclass

from transformers import MllamaForConditionalGeneration, AutoProcessor
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

class LlamaCaptionWorker(BaseWorker):
    def initialize_worker(self):
        """Initialize the Llama model and processor"""
        gpu_id = self.worker_params['gpu_id']
        torch.cuda.set_device(gpu_id)
        
        MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.model = MllamaForConditionalGeneration.from_pretrained(
            MODEL, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        )
        self.model.to(f'cuda:{gpu_id}')
        self.processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        self.prompt = self.worker_params['prompt']

    def process_item(self, item: BatchItem) -> Tuple[List[int], List[str]]:
        """Process a batch of images and return their captions"""
        pilimages_batch = []
        for path in item.image_paths:
            img = Image.open(path).convert('RGB')
            pilimages_batch.append(img)

        text_batch = []
        for img in pilimages_batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        { "type": "image" },
                        {"type": "text", "text": self.prompt },
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages)
            text_batch.append(text)

        inputs = self.processor(pilimages_batch, text_batch, return_tensors="pt")
        inputs.to(f'cuda:{self.worker_params["gpu_id"]}')

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=300, 
                temperature=0.5
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return item.image_indices, [caption.removeprefix("assistant\n\n").strip() for caption in output_text]

    def shutdown_worker(self):
        """Clean up GPU resources"""
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()


class LlamaCaption(Node):
    '''Generates image captions using the Llama 3.2 Vision 11B model'''
    DEFAULT_PROMPT = "Write 2-3 sentences that describe this image"

    def __init__(self,
                 target_prop: str = 'caption',
                 prompt: Optional[str] = None,
                 instructions: Optional[str] = None,
                 batch_size: int = 1):
        '''Initializes the LlamaCaption node

        Args:
            target_prop (str): The property to store the caption in
            prompt (str): The prompt to use for the Llama model (read the code)
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
            image._llama_caption = CachedProperty('llama-3.2-11B-vision', 'v1', self.prompt, image)
            setattr(image, self.target_prop, ComputedProperty(lambda image: image._llama_caption.value if image._llama_caption.has_value else None))
            if not image._llama_caption.has_value:
                needs_caption.append(image)

        if len(needs_caption) == 0:
            self.log.info("All images already have captions, skipping")
            return dataset

        self.log.info(f"Generating captions for {len(needs_caption)} images")

        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
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
        controller = ParallelController(LlamaCaptionWorker, worker_params_list)
        
        with tqdm(total=len(needs_caption), desc="Llama Vision") as pbar:
            for success, result in controller.run(batches):
                if not success:
                    raise RuntimeError(f"Worker failed: {result}")
                
                indices, captions = result
                for idx, caption in zip(indices, captions):
                    needs_caption[idx]._llama_caption.value = caption
                    self.log.info(f"{self.workspace.get_path(needs_caption[idx])} => {caption}")
                pbar.update(len(indices))
        
        return dataset

__all__ = ['LlamaCaption']

