# 
# This file is based on code and models first published at:
# 
#     https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one/blob/main/app.py
#
    
from beprepared.workspace import Workspace
from beprepared.node import Node
from beprepared.properties import CachedProperty, ComputedProperty

from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import torchvision.transforms.functional as TVF
from typing import Literal, Union
from tqdm import tqdm

import base64
import openai

CLIP_PATH = "google/siglip-so400m-patch14-384"
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B"

CAPTION_TYPE_MAP = {
    ("descriptive", "formal", False, False): ["Write a descriptive caption for this image in a formal tone."],
    ("descriptive", "formal", False, True): ["Write a descriptive caption for this image in a formal tone within {word_count} words."],
    ("descriptive", "formal", True, False): ["Write a {length} descriptive caption for this image in a formal tone."],
    ("descriptive", "informal", False, False): ["Write a descriptive caption for this image in a casual tone."],
    ("descriptive", "informal", False, True): ["Write a descriptive caption for this image in a casual tone within {word_count} words."],
    ("descriptive", "informal", True, False): ["Write a {length} descriptive caption for this image in a casual tone."],
    ("stable_diffusion", "formal", False, False): ["Write a stable diffusion prompt for this image."],
    ("stable_diffusion", "formal", False, True): ["Write a stable diffusion prompt for this image within {word_count} words."],
    ("stable_diffusion", "formal", True, False): ["Write a {length} stable diffusion prompt for this image."],
    ("booru", "formal", False, False): ["Write a list of Booru tags for this image."],
    ("booru", "formal", False, True): ["Write a list of Booru tags for this image within {word_count} words."],
    ("booru", "formal", True, False): ["Write a {length} list of Booru tags for this image."],
}

class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int, deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))

        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.concat((
                vision_outputs[-2],
                vision_outputs[3],
                vision_outputs[7],
                vision_outputs[13],
                vision_outputs[20],
            ), dim=-1)
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"
            assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        other_tokens = self.other_tokens(torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
        assert other_tokens.shape == (x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

class JoyCaptionAlphaOne(Node):
    def __init__(self,
                 target_prop: str = 'caption',
                 caption_type: Literal['descriptive', 'stable_diffusion', 'booru'] = 'descriptive',
                 caption_tone: Literal['formal', 'informal'] = 'formal',
                 caption_length: Union[int, 'any'] = 'any',
                 batch_size: int = 4) -> None:
        super().__init__()
        self.target_prop = target_prop
        self.caption_type = caption_type
        self.caption_tone = caption_tone
        self.caption_length = caption_length
        self.batch_size = batch_size

        self.client = openai.AsyncClient(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def eval(self, dataset):
        needs_caption = []
        for image in dataset.images:
            image._joycaption = CachedProperty('joycaption', 'v1', self.caption_type, self.caption_tone, self.caption_length, image)
            setattr(image, self.target_prop, ComputedProperty(lambda image: image._joycaption.value if image._joycaption.has_value else None))
            if not image._joycaption.has_value:
                needs_caption.append(image)

        if len(needs_caption) == 0:
            self.log.info(f"All images already have captions, skipping")
            return dataset

        self.log.info(f"Generating captions for {len(needs_caption)} images")

        cachedir = self.workspace.cache.dir('joycaption-alpha-one')
        image_adapter_path = cachedir.download(
            "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one/resolve/main/9em124t2-499968/image_adapter.pt",
            "image_adapter.pt"
        )
        clip_model_path = cachedir.download(
            "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one/resolve/main/9em124t2-499968/clip_model.pt",
            "clip_model.pt"
        )
        text_model_path = os.path.join(cachedir.dir, 'text_model')
        text_adapter_config_path = cachedir.download(
            "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one/resolve/main/9em124t2-499968/text_model/adapter_config.json",
            "text_model/adapter_config.json"
        )
        cachedir.download(
            "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one/resolve/main/9em124t2-499968/text_model/adapter_model.safetensors",
            "text_model/adapter_model.safetensors"
        )

        self.log.info("Loading CLIP")
        self.clip_model = AutoModel.from_pretrained(CLIP_PATH)
        self.clip_model = self.clip_model.vision_model

        self.log.info("Replacing CLIP Layers with JoyCaption Finetune")
        checkpoint = torch.load(clip_model_path, map_location='cpu', weights_only=True)
        checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
        self.clip_model.load_state_dict(checkpoint)
        del checkpoint
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.clip_model.to("cuda")

        self.log.info(f"Loading LLama 3.1-8B with JoyCaption Lora")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        assert isinstance(self.tokenizer, PreTrainedTokenizer) or isinstance(self.tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(self.tokenizer)}"
        self.text_model = AutoModelForCausalLM.from_pretrained(text_model_path, torch_dtype=torch.bfloat16)
        self.text_model.eval()
        self.text_model.to("cuda")

        self.log.info("Loading JoyCaption image adapter")
        self.image_adapter = ImageAdapter(self.clip_model.config.hidden_size, self.text_model.config.hidden_size, False, False, 38, False)
        self.image_adapter.load_state_dict(torch.load(image_adapter_path, map_location='cpu', weights_only=True))
        self.image_adapter.eval()
        self.image_adapter.to("cuda")

        batches = [needs_caption[i:i + self.batch_size] for i in range(0, len(needs_caption), self.batch_size)]

        for batch in tqdm(batches, desc="JoyCaption"):
            input_images = [Image.open(self.workspace.get_path(image)) for image in batch]
            captions = self.generate_batch(input_images, self.caption_type, self.caption_tone, self.caption_length)
            for image, caption in zip(batch, captions):
                clean_caption = caption.rstrip('!')
                image._joycaption.value = clean_caption
                self.log.info(f"{self.workspace.get_path(image)} => {clean_caption}")

        return dataset

    def generate_batch(self, input_images: list, caption_type: str, caption_tone: str, caption_length: Union[str, int]) -> list:
        torch.cuda.empty_cache()

        length = None if caption_length == "any" else caption_length

        if caption_type in ["booru", "stable_diffusion"]:
            caption_tone = "formal"

        prompts = []
        for _ in input_images:
            prompt_key = (caption_type, caption_tone, isinstance(length, str), isinstance(length, int))
            if prompt_key not in CAPTION_TYPE_MAP:
                raise ValueError(f"Invalid caption type: {prompt_key}")
            prompt_str = CAPTION_TYPE_MAP[prompt_key][0].format(length=length, word_count=length)
            prompts.append(prompt_str)

        processed_images = []
        for image in input_images:
            img = image.resize((384, 384), Image.LANCZOS)
            pixel_values = TVF.pil_to_tensor(img).unsqueeze(0) / 255.0
            pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
            pixel_values = pixel_values.to('cuda')
            processed_images.append(pixel_values)

        pixel_values = torch.cat(processed_images, dim=0)
        pixel_values = pixel_values.to('cuda')

        tokenized_prompts = [self.tokenizer.encode(p, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False) for p in prompts]
        prompt_ids = torch.cat(tokenized_prompts, dim=0).to('cuda')

        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = self.clip_model(pixel_values=pixel_values, output_hidden_states=True)
            image_features = vision_outputs.hidden_states
            embedded_images = self.image_adapter(image_features)
            embedded_images = embedded_images.to('cuda')

        prompt_embeds = self.text_model.model.embed_tokens(prompt_ids)
        embedded_bos = self.text_model.model.embed_tokens(torch.tensor([self.tokenizer.bos_token_id], device=self.text_model.device, dtype=torch.int64))
        eot_embed = self.image_adapter.get_eot_embedding().unsqueeze(0).to(dtype=self.text_model.dtype)

        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            prompt_embeds,
            eot_embed.expand(embedded_images.shape[0], -1, -1),
        ], dim=1)

        input_ids = torch.cat([
            torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long, device='cuda').unsqueeze(0).expand(len(input_images), -1),
            torch.zeros((len(input_images), embedded_images.shape[1]), dtype=torch.long, device='cuda'),
            prompt_ids,
            torch.tensor([self.tokenizer.convert_tokens_to_ids("<|eot_id|>")], dtype=torch.long, device='cuda').unsqueeze(0).expand(len(input_images), -1),
        ], dim=1)
        attention_mask = torch.ones_like(input_ids)

        generate_ids = self.text_model.generate(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            suppress_tokens=None,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generate_ids = generate_ids[:, input_ids.shape[1]:]
        generate_ids = torch.where(
            generate_ids == self.tokenizer.eos_token_id,
            torch.zeros_like(generate_ids),
            generate_ids
        )

        captions = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        captions = [caption.strip() for caption in captions]
        return captions

__all__ = [ 'JoyCaptionAlphaOne' ]

