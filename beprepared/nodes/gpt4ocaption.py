import openai
import os
import base64
import asyncio
from pydantic import BaseModel
from typing import Optional, Literal
from tqdm.asyncio import tqdm

from beprepared.workspace import Workspace
from beprepared.node import Node 
from beprepared.properties import CachedProperty, ComputedProperty
from beprepared.nodes.convert_format import convert_image

class GPT4oCaption(Node):
    '''Generates captions for images using GPT-4o'''

    DEFAULT_PROMPT = """
Please write a detailed caption for this image. Describe the main subjects, 
their characteristics, positions, and any notable aspects of the background, 
lighting, style, and quality. 

If you are unable to assist with this request, please write "Caption not available" and nothing else.

Your output should be ONLY the caption, with no boilerplate, pleasantries, or other details.
""".strip()

    def __init__(self,
                 target_prop: str = 'caption',
                 prompt: Optional[str] = None,
                 instructions: Optional[str] = None,
                 parallel: int = 8):
        '''Initializes the GPT4oCaption node

        Args:
            target_prop (str): The property to store the caption in
            prompt (str): The prompt to use for the GPT-4o model
            instructions (str): Additional instructions to include in the prompt
            parallel (int): The number of images to process in parallel
        '''
        super().__init__()

        self.parallel = parallel

        self.client = openai.AsyncClient(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        self.prompt = prompt or self.DEFAULT_PROMPT
        if instructions:
            self.prompt = f"{self.prompt}\n\n{instructions}"
        self.target_prop = target_prop

    def eval(self, dataset):
        needs_caption = []
        for image in dataset.images:
            image._gpt4o_caption = CachedProperty('gpt4o-caption', 'v2', self.prompt, image)
            setattr(image, self.target_prop, ComputedProperty(lambda image: image._gpt4o_caption.value if image._gpt4o_caption.has_value else None))
            if not image._gpt4o_caption.has_value:
                needs_caption.append(image)

        if len(needs_caption) == 0:
            self.log.info(f"All images already have captions, skipping")
            return dataset

        asyncio.run(self._eval_parallel(needs_caption))

        return dataset

    async def _eval_parallel(self, needs_caption):
        semaphore = asyncio.Semaphore(self.parallel)
        tasks = [self._process_image_async(image, semaphore) for image in needs_caption]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="GPT4o Captioning"):
            await f

    async def _process_image_async(self, image, semaphore):
        async with semaphore:
            path = self.workspace.get_path(image)
            if image.format == 'PNG':
                base64_image = base64.b64encode(open(path, "rb").read()).decode("utf-8")
                base64_url = f"data:image/png;base64,{base64_image}"
            elif image.format == 'JPEG':
                base64_image = base64.b64encode(open(path, "rb").read()).decode("utf-8")
                base64_url = f"data:image/jpeg;base64,{base64_image}"
            else:
                bytes_data = convert_image(path, 'PNG')
                base64_image = base64.b64encode(bytes_data).decode("utf-8")
                base64_url = f"data:image/png;base64,{base64_image}"

            response = await self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.prompt.strip()
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_url
                                },
                            },
                        ]
                    },
                ],
                temperature=0.6,
                max_tokens=300,
            )

            caption = response.choices[0].message.content.strip()
            image._gpt4o_caption.value = caption if "Caption not available" not in caption else None
            self.log.info(f"{path} => {caption}")

