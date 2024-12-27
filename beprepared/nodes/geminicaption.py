import os
import base64
import asyncio
from google.api_core import exceptions
from pydantic import BaseModel
from typing import Optional, Literal
from tqdm.asyncio import tqdm
import google.generativeai as genai

from beprepared.workspace import Workspace
from beprepared.node import Node 
from beprepared.properties import CachedProperty, ComputedProperty
from beprepared.nodes.convert_format import convert_image

class GeminiCaptionResult(BaseModel):
    caption: Optional[str]

class GeminiCaption(Node):
    '''Generates captions for images using Gemini 2.0 Flash Vision'''

    DEFAULT_PROMPT = """
Please write a detailed caption for this image. Describe the main subjects, 
their characteristics, positions, and any notable aspects of the background, 
lighting, style, and quality. 

Your output should be ONLY the caption, with no boilerplate, pleasantries, or other details.
""".strip()

    def __init__(self,
                 target_prop: str = 'caption',
                 prompt: Optional[str] = None,
                 instructions: Optional[str] = None,
                 parallel: int = 2):
        '''Initializes the GeminiCaption node

        Args:
            target_prop (str): The property to store the caption in
            prompt (str): The prompt to use for the Gemini model
            instructions (str): Additional instructions to include in the prompt
            parallel (int): The number of images to process in parallel
        '''
        super().__init__()
        
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Either GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set")
            
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel('gemini-2.0-flash-exp')

        self.parallel = parallel
        self.prompt = prompt or self.DEFAULT_PROMPT
        if instructions:
            self.prompt = f"{self.prompt}\n\n{instructions}"
        self.target_prop = target_prop

    def eval(self, dataset):
        needs_caption = []
        for image in dataset.images:
            image._gemini_caption = CachedProperty('gemini-2.0-flash', 'v1', self.prompt, image)
            setattr(image, self.target_prop, ComputedProperty(lambda image: image._gemini_caption.value if image._gemini_caption.has_value else None))
            if not image._gemini_caption.has_value:
                needs_caption.append(image)

        if len(needs_caption) == 0:
            self.log.info(f"All images already have captions, skipping")
            return dataset

        asyncio.run(self._eval_parallel(needs_caption))

        return dataset

    async def _eval_parallel(self, needs_caption):
        semaphore = asyncio.Semaphore(self.parallel)
        tasks = [self._process_image_async(image, semaphore) for image in needs_caption]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Gemini Captioning"):
            await f

    async def _process_image_async(self, image, semaphore):
        async with semaphore:
            path = self.workspace.get_path(image)
            
            # Convert image to base64
            if image.format == 'PNG':
                mime_type = 'image/png'
                base64_image = base64.b64encode(open(path, "rb").read()).decode("utf-8")
            elif image.format == 'JPEG':
                mime_type = 'image/jpeg'
                base64_image = base64.b64encode(open(path, "rb").read()).decode("utf-8")
            else:
                mime_type = 'image/png'
                bytes_data = convert_image(path, 'PNG')
                base64_image = base64.b64encode(bytes_data).decode("utf-8")

            # Generate caption with exponential backoff
            delay = 2
            max_retries = 10
            attempt = 0
            
            while True:
                try:
                    response = await self.client.generate_content_async(
                        contents=[
                            {
                                "role": "user",
                                "parts": [
                                    {"text": self.prompt.strip()},
                                    {
                                        "inline_data": {
                                            "mime_type": mime_type,
                                            "data": base64_image
                                        }
                                    }
                                ]
                            }
                        ]
                    )
                    break
                except exceptions.ResourceExhausted:
                    attempt += 1
                    if attempt >= max_retries:
                        raise
                    self.log.warning(f"Hit quota limit, backing off for {delay}s before retry {attempt}/{max_retries}")
                    await asyncio.sleep(delay)
                    delay *= 2

            try:
                result = GeminiCaptionResult(caption=response.text)
                image._gemini_caption.value = result
                self.log.info(f"{path} => {result.caption}")
            except ValueError as e:
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    self.log.error(f"Gemini blocked prompt for {path} due to: {response.prompt_feedback.block_reason}")
                    image._gemini_caption.value = GeminiCaptionResult(caption=None)
