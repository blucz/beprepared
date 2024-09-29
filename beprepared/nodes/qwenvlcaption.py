import os
from typing import Optional, Literal

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from tqdm import tqdm
from PIL import Image

from beprepared.workspace import Workspace
from beprepared.node import Node 
from beprepared.properties import CachedProperty, ComputedProperty
from beprepared.nodes.convert_format import convert_image
from qwen_vl_utils import process_vision_info

class QwenVLCaption(Node):
    DEFAULT_PROMPT = """Describe the contents and style of this image."""

    def __init__(self,
                 target_prop:    str                              = 'caption',
                 prompt:         Optional[str] = None,
                 instructions:   Optional[str] = None,
                 batch_size:     int           = 3):
        super().__init__()
        self.target_prop  = target_prop
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.batch_size = batch_size
        if instructions:
            self.prompt = f"{self.prompt}\n\n{instructions}"

    def eval(self, dataset):
        needs_caption = []
        for image in dataset.images:
            image._qwenvl_caption = CachedProperty('qwen-vl', 'v1', self.prompt, image)
            setattr(image, self.target_prop, ComputedProperty(lambda image: image._qwenvl_caption.value if image._qwenvl_caption.has_value else None))
            if not image._qwenvl_caption.has_value:
                needs_caption.append(image)

        if len(needs_caption) == 0: 
            self.log.info("All images already have captions, skipping")
            return dataset

        MODEL = "Qwen/Qwen2-VL-7B-Instruct"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL, 
                torch_dtype=torch.bfloat16,
                trust_remote_code=False,
        )
        model.to('cuda')
        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

        for i in tqdm(range(0, len(needs_caption), self.batch_size), desc="Qwen-VL"):
            batch_images = needs_caption[i:i + self.batch_size]
            pil_images = []
            for image in batch_images:
                path = self.workspace.get_path(image)
                image = Image.open(path).convert('RGB')
                pil_images.append(image)

            messages_batch = []
            text_batch = []
            for img in pil_images:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            { "type": "image", "image": img, },
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ]
                messages_batch.append(messages)
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                text_batch.append(text)

            image_inputs, video_inputs = process_vision_info(messages_batch)
            inputs = processor(
                text=text_batch,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=300, temperature=0.5)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for image, caption in zip(batch_images, output_text):
                image._qwenvl_caption.value = caption.strip()

        return dataset

