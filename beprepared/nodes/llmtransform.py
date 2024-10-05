from beprepared.node import Node 
from beprepared.image import Image
from beprepared.properties import CachedProperty

from typing import Callable
import asyncio
from litellm import acompletion

class LLMCaptionTransform(Node):
    '''Transforms image captions using a language model'''
    def __init__(self, model: str, prompt: Callable[[Image], str], target_prop: str, parallel: int = 8):
        '''Initializes the LLMCaptionTransform node

        Args:
            model (str): The name of the language model to use
            prompt (Callable[[Image], str]): A function that takes an image and returns a prompt for the language model
            target_prop (str): The property to store the transformed caption in (default is 'caption')
            parallel (int): The number of images to process in parallel
        '''
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.target_prop = target_prop
        self.parallel = parallel

    def eval(self, dataset):
        needs_caption = []
        for image in dataset.images:
            prompt = self.prompt(image)
            prop = CachedProperty(self.model, prompt, image)
            setattr(image, self.target_prop, prop)
            if not prop.has_value:
                needs_caption.append((image, prop, prompt))

        if len(needs_caption) == 0:
            self.log.info("All images already have transformed captions, skipping")
            return dataset

        self.log.info(f"Transforming captions using LLM for {len(needs_caption)} images")

        async def transform_captions():
            semaphore = asyncio.Semaphore(self.parallel)

            async def process_image(image, prop, prompt):
                async with semaphore:
                    messages = [{"content": prompt, "role": "user"}]
                    response = await acompletion(model=self.model, messages=messages)
                    content = response['choices'][0]['message']['content']
                    prop.value = content
                    self.log.debug(f"Transformed caption for image {image.id}: {content}")

            tasks = [
                process_image(image, prop, prompt) 
                for image, prop, prompt in needs_caption
            ]

            await asyncio.gather(*tasks)

        asyncio.run(transform_captions())

        self.log.info("Completed transforming captions.")

        return dataset

