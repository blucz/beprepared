from beprepared.node import Node 
from beprepared.image import Image
from beprepared.properties import CachedProperty

from typing import Callable
import asyncio
from litellm import acompletion
from tqdm import tqdm

class LLMCaptionTransform(Node):
    '''Transforms image captions using a language model'''
    def __init__(self, model: str, prompt: Callable[[Image], str], target_prop: str, parallel: int = 20, temperature = 0.5):
        '''Initializes the LLMCaptionTransform node

        Args:
            model (str): The name of the language model to use
            prompt (Callable[[Image], str]): A function that takes an image and returns a prompt for the language model
            target_prop (str): The property to store the transformed caption in (default is 'caption')
            parallel (int): The number of images to process in parallel
            temperature (float): The temperature to use when sampling from the language model
        '''
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.target_prop = target_prop
        self.parallel = parallel
        self.temperature = temperature

    def eval(self, dataset):
        needs_caption = []

        params = {
            'temperature': self.temperature
        }

        for image in dataset.images:
            prompt = self.prompt(image)
            prop = CachedProperty('llm', self.model, params, prompt)
            setattr(image, self.target_prop, prop)
            if not prop.has_value:
                needs_caption.append((image, prop, prompt))

        if len(needs_caption) == 0:
            self.log.info("All images already have transformed captions, skipping")
            return dataset

        self.log.info(f"Transforming captions using LLM for {len(needs_caption)} images")

        async def transform_captions():
            semaphore = asyncio.Semaphore(self.parallel)
            total = len(needs_caption)

            with tqdm(total=total, desc="Processing Captions") as pbar:
                async def process_image(image, prop, prompt):
                    async with semaphore:
                        messages = [{"content": prompt, "role": "user"}]
                        response = await acompletion(model=self.model, messages=messages, temperature=self.temperature)
                        content = response['choices'][0]['message']['content']
                        prop.value = content
                        self.log.debug(f"Transformed caption for image {image.id}: {content}")
                        pbar.update(1)

                tasks = [
                    process_image(image, prop, prompt) 
                    for image, prop, prompt in needs_caption
                ]

                await asyncio.gather(*tasks)

        asyncio.run(transform_captions())

        self.log.info("Completed transforming captions.")

        return dataset

