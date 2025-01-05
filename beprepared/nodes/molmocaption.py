import gc
from typing import Optional

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch
from .utils import tqdm
from PIL import Image

from beprepared.node import Node 
from beprepared.properties import CachedProperty, ComputedProperty

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

        self.log.info(f"Generating captions for {len(needs_caption)} images")

        MODEL = "allenai/Molmo-7B-D-0924"
        
        # Load processor and model
        processor = AutoProcessor.from_pretrained(
            MODEL,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to('cuda')

        for i in tqdm(range(0, len(needs_caption), self.batch_size), desc="Molmo"):
            batch_images = needs_caption[i:i + self.batch_size]

            # Prepare images and text inputs
            pil_images = []
            for image in batch_images:
                path = self.workspace.get_path(image)
                img = Image.open(path).convert('RGB')
                pil_images.append(img)

            # Process inputs
            inputs = processor.process(
                images=pil_images,
                text=self.prompt
            )
            
            # Move inputs to device and batch
            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

            # Generate captions
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )

            # Decode and store captions
            for idx, image in enumerate(batch_images):
                generated_tokens = output[idx, inputs['input_ids'].size(1):]
                caption = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                image._molmo_caption.value = caption.strip()
                self.log.info(f"{self.workspace.get_path(image)} => {caption}")

        # Cleanup
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        
        return dataset

__all__ = ['MolmoCaption']
