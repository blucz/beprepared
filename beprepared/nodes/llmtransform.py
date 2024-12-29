from beprepared.node import Node 
from beprepared.image import Image
from beprepared.properties import CachedProperty

from typing import Callable
import asyncio
from litellm import acompletion
from .utils import tqdm

class LLMCaptionVariations(Node):
    '''Generates variations of image captions using LLaMA 3.1 8B Instruct'''
    def __init__(self, target_prop: str = 'caption', variations: int = 2, parallel: int = 20, temperature: float = 0.7,
                 model: str = "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
        '''Initializes the LLMCaptionVariations node

        Args:
            target_prop (str): Property name to store the caption in
            variations (int): Number of variations to generate per image (creates additional images)
            parallel (int): The number of images to process in parallel
            temperature (float): The temperature to use when sampling from the model
            model (str): The LLM model to use for generating variations
        '''
        super().__init__()
        self.model = model
        self.target_prop = target_prop
        self.variations = variations
        self.parallel = parallel
        self.temperature = temperature

    def eval(self, dataset):
        original_images = dataset.images.copy()
        new_images = []
        needs_variations = []

        # Keep track of which variations we need to generate
        for image in original_images:
            caption = getattr(image, self.target_prop).value
            if caption:
                for i in range(self.variations):
                    new_image = image.copy()
                    params = {
                        'temperature': self.temperature,
                        'variation_index': i
                    }
                    prop = CachedProperty('llm', f"{self.model}", params, caption)
                    setattr(new_image, self.target_prop, prop)
                    new_images.append(new_image)
                    if not prop.has_value:
                        needs_variations.append((new_image, prop, caption))

        if len(needs_variations) == 0:
            self.log.info("All variations already generated, skipping")
            return dataset

        self.log.info(f"Generating {len(needs_variations)} caption variations")

        async def generate_variations():
            semaphore = asyncio.Semaphore(self.parallel)
            total = len(needs_variations)

            with tqdm(total=total, desc="Generating Variations") as pbar:
                async def process_variation(image, prop, caption):
                    async with semaphore:
                        prompt = f"""Generate a different way to describe this image that captures the same key information but uses different phrasing and word choices. 

                        Original caption: {caption}

                        Your output should be the caption only, with no extra text."""
                        messages = [{"content": prompt, "role": "user"}]
                        response = await acompletion(model=self.model, messages=messages, temperature=self.temperature)
                        content = response['choices'][0]['message']['content']
                        prop.value = content
                        self.log.debug(f"Generated variation for image {image.id}: {content}")
                        pbar.update(1)

                tasks = [
                    process_variation(image, prop, caption)
                    for image, prop, caption in needs_variations
                ]

                await asyncio.gather(*tasks)

        asyncio.run(generate_variations())

        self.log.info("Completed generating caption variations.")
        
        # Add the new images to the dataset
        self.log.info(f"added {len(new_images)} new images to the dataset")
        dataset.images.extend(new_images)

        return dataset

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

