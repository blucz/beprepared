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

class GPT4oCaptionResult(BaseModel):
    booru: str
    description: str

class GPT4oCaption(Node):
    '''Generates captions for images using GPT-4o'''

    DEFAULT_PROMPT = """
Your task is to write a caption for this image in two styles:

(1) A list of "booru" tags, where elements of the image are listed in a simple and terse fashion. The goal is to capture all items in the image. For reference, the most common booru tags include: 1girl, solo, highres, long_hair, commentary_request, breasts, looking_at_viewer, blush, smile, open_mouth, short_hair, blue_eyes, simple_background, shirt, absurdres, large_breasts, skirt, blonde_hair, multiple_girls, brown_hair, black_hair, long_sleeves, hair_ornament, white_background, 1boy, gloves, red_eyes, bad_id, dress, thighhighs, hat, holding, commentary, original, bow, navel, animal_ears, ribbon, hair_between_eyes, closed_mouth, 2girls, bad_pixiv_id, touhou, cleavage, jewelry, bare_shoulders, very_long_hair, sitting, twintails, medium_breasts, brown_eyes, standing, nipples, green_eyes, photoshop_, underwear, blue_hair, jacket, school_uniform, purple_eyes, collarbone, tail, white_shirt, full_body, upper_body, panties, closed_eyes, swimsuit, yellow_eyes, white_hair, pink_hair, monochrome, grey_hair, ahoge, hair_ribbon, braid, purple_hair, translated, male_focus, ponytail.

(2) A 2-4 sentence description formatted in one paragraph that explains the context of the image. 

In both cases, be sure to clearly describe any subjects of the image and their characteristics and position within the image. Also describe aspects of the background, lighting, style, and quality if those details are apparent.
""".strip()

    def __init__(self,
                 target_prop: str = 'caption',
                 caption_type: Literal['descriptive', 'booru'] = 'descriptive',
                 prompt: Optional[str] = None,
                 instructions: Optional[str] = None,
                 parallel: int = 8):
        '''Initializes the GPT4oCaption node

        Args:
            target_prop (str): The property to store the caption in
            caption_type (str): The type of caption to generate ('descriptive' or 'booru')
            prompt (str): The prompt to use for the GPT-4o model (read the code)
            instructions (str): Additional instructions to include in the built-in prompt
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
        self.caption_type = caption_type

    def eval(self, dataset):
        needs_caption = []
        for image in dataset.images:
            image._gpt4o_caption = CachedProperty('gpt4o-caption', 'v1', self.prompt, image)
            def compute_target_prop(image):
                if not image._gpt4o_caption.has_value:
                    return None 
                elif self.caption_type == 'descriptive':
                    return image._gpt4o_caption.value.description
                elif self.caption_type == 'booru':
                    return image._gpt4o_caption.value.booru
                else:
                    return None
            setattr(image, self.target_prop, ComputedProperty(compute_target_prop))
            target_prop = getattr(image, self.target_prop)
            if not target_prop.value:
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
                response_format=GPT4oCaptionResult
            )

            result = response.choices[0].message.parsed
            image._gpt4o_caption.value = result
            self.log.info(f"{path} => {getattr(image, self.target_prop).value}")

