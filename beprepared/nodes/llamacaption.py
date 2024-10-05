from typing import Optional

from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
from tqdm import tqdm
from PIL import Image

from beprepared.node import Node 
from beprepared.properties import CachedProperty, ComputedProperty

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

        MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        model = MllamaForConditionalGeneration.from_pretrained(
                MODEL, 
                torch_dtype=torch.bfloat16,
                trust_remote_code=False,
        )
        model.to('cuda')
        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

        for i in tqdm(range(0, len(needs_caption), self.batch_size), desc="Llama Vision"):
            batch_images = needs_caption[i:i + self.batch_size]

            pilimages_batch = []
            for image in batch_images:
                path = self.workspace.get_path(image)
                image = Image.open(path).convert('RGB')
                pilimages_batch.append(image)

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
                text = processor.apply_chat_template(messages)
                text_batch.append(text)

            inputs = processor(pilimages_batch, text_batch, return_tensors="pt")
            inputs.to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=300, 
                    temperature=0.5
                )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for image, caption in zip(batch_images, output_text):
                clean_caption = caption.removeprefix("assistant\n\n").strip()
                image._llama_caption.value = clean_caption
                self.log.info(f"{self.workspace.get_path(image)} => {clean_caption}")

        return dataset

__all__ = ['LlamaCaption']

