# 
# This file is based on code and models first published at:
# 
#     https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one/blob/main/app.py
#
    
from beprepared.workspace import Workspace, Abort
from beprepared.node import Node
from beprepared.properties import CachedProperty, ComputedProperty
from dataclasses import dataclass
from typing import List, Tuple

from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import torchvision.transforms.functional as TVF
from typing import Literal, Union
from .utils import tqdm
from .parallelworker import ParallelController, BaseWorker

@dataclass
class BatchItem:
    """Represents a batch of images to process"""
    image_indices: List[int]  # Original indices in the dataset
    image_paths: List[str]    # Paths to the images

import base64
import openai
import gc

CLIP_PATH = "google/siglip-so400m-patch14-384"
BASE_MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B"

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

class JoyCaptionAlphaOneWorker(BaseWorker):
    def initialize_worker(self):
        """Initialize the models and adapters"""
        gpu_id = self.worker_params['gpu_id']
        torch.cuda.set_device(gpu_id)
        
        cachedir = self.worker_params['cachedir'].dir
        image_adapter_path = os.path.join(cachedir, "image_adapter.pt")
        clip_model_path = os.path.join(cachedir, "clip_model.pt")
        text_model_path = os.path.join(cachedir, 'text_model')

        self.clip_model = AutoModel.from_pretrained(CLIP_PATH)
        self.clip_model = self.clip_model.vision_model

        checkpoint = torch.load(clip_model_path, map_location='cpu', weights_only=True)
        checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
        self.clip_model.load_state_dict(checkpoint)
        del checkpoint
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.clip_model.to(f'cuda:{gpu_id}')

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)

        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_model_path, 
            device_map=gpu_id,
            torch_dtype=torch.bfloat16
        )
        self.text_model.eval()

        self.image_adapter = ImageAdapter(
            self.clip_model.config.hidden_size,
            self.text_model.config.hidden_size,
            False, False, 38, False
        )
        self.image_adapter.load_state_dict(
            torch.load(image_adapter_path, map_location='cpu', weights_only=True)
        )
        self.image_adapter.eval()
        self.image_adapter.to(f'cuda:{gpu_id}')

        # Store parameters
        self.caption_type = self.worker_params['caption_type']
        self.caption_tone = self.worker_params['caption_tone']
        self.caption_length = self.worker_params['caption_length']

    def process_item(self, item: BatchItem) -> Tuple[List[int], List[str]]:
        """Process a batch of images and return their captions"""
        input_images = [Image.open(path) for path in item.image_paths]
        
        torch.cuda.empty_cache()
        length = None if self.caption_length == "any" else self.caption_length

        if self.caption_type in ["booru", "stable_diffusion"]:
            caption_tone = "formal"
        else:
            caption_tone = self.caption_tone

        prompts = []
        for _ in input_images:
            prompt_key = (self.caption_type, caption_tone, isinstance(length, str), isinstance(length, int))
            if prompt_key not in CAPTION_TYPE_MAP:
                raise ValueError(f"Invalid caption type: {prompt_key}")
            prompt_str = CAPTION_TYPE_MAP[prompt_key][0].format(length=length, word_count=length)
            prompts.append(prompt_str)

        processed_images = []
        for image in input_images:
            img = image.resize((384, 384), Image.LANCZOS)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            pixel_values = TVF.pil_to_tensor(img).unsqueeze(0) / 255.0
            pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
            pixel_values = pixel_values.to(f'cuda:{self.worker_params["gpu_id"]}')
            processed_images.append(pixel_values)

        pixel_values = torch.cat(processed_images, dim=0)
        pixel_values = pixel_values.to(f'cuda:{self.worker_params["gpu_id"]}')

        tokenized_prompts = [
            self.tokenizer.encode(p, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)
            for p in prompts
        ]
        prompt_ids = torch.cat(tokenized_prompts, dim=0).to(f'cuda:{self.worker_params["gpu_id"]}')

        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = self.clip_model(pixel_values=pixel_values, output_hidden_states=True)
            image_features = vision_outputs.hidden_states
            embedded_images = self.image_adapter(image_features)
            embedded_images = embedded_images.to(f'cuda:{self.worker_params["gpu_id"]}')

        prompt_embeds = self.text_model.model.embed_tokens(prompt_ids)
        embedded_bos = self.text_model.model.embed_tokens(
            torch.tensor([self.tokenizer.bos_token_id], device=f'cuda:{self.worker_params["gpu_id"]}', dtype=torch.int64)
        )
        eot_embed = self.image_adapter.get_eot_embedding().unsqueeze(0).to(dtype=self.text_model.dtype)

        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            prompt_embeds,
            eot_embed.expand(embedded_images.shape[0], -1, -1),
        ], dim=1)

        input_ids = torch.cat([
            torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long, device=f'cuda:{self.worker_params["gpu_id"]}')
                .unsqueeze(0).expand(len(input_images), -1),
            torch.zeros(
                (len(input_images), embedded_images.shape[1]),
                dtype=torch.long,
                device=f'cuda:{self.worker_params["gpu_id"]}'
            ),
            prompt_ids,
            torch.tensor([self.tokenizer.convert_tokens_to_ids("<|eot_id|>")], dtype=torch.long, device=f'cuda:{self.worker_params["gpu_id"]}')
                .unsqueeze(0).expand(len(input_images), -1),
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

        captions = self.tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=False, 
            clean_up_tokenization_spaces=False
        )
        clean_captions = [caption.strip().rstrip('!') for caption in captions]

        return item.image_indices, clean_captions

    def shutdown_worker(self):
        """Clean up GPU resources"""
        del self.clip_model
        del self.tokenizer
        del self.text_model
        del self.image_adapter
        gc.collect()
        torch.cuda.empty_cache()


class JoyCaptionAlphaOne(Node):
    '''Generates captions for images using the JoyCaption Alpha One model.

    For more information on JoyCaption, see https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-one.'''
    def __init__(self,
                 target_prop: str = 'caption',
                 caption_type: Literal['descriptive', 'stable_diffusion', 'booru'] = 'descriptive',
                 caption_tone: Literal['formal', 'informal'] = 'formal',
                 caption_length: Union[int, 'any'] = 'any',
                 batch_size: int = 4) -> None:
        '''Initializes the JoyCaptionAlphaOne node

        Args:
            target_prop (str): The property to store the caption in (default is 'caption')
            caption_type (str): The type of caption to generate ('descriptive', 'stable_diffusion', 'booru')
            caption_tone (str): The tone of the caption ('formal' or 'informal')
            caption_length (int | 'any'): The length of the caption, or 'any' for any length
            batch_size (int): The number of images to process in parallel. If you are running out of memory, try reducing this value. Default is 4.
        '''
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

        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        self.log.info(f"Processing {len(needs_caption)} images with JoyCaptionAlphaOne using {num_gpus} GPUs")
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available")

        # Create worker parameters for each GPU
        worker_params_list = [
            {
                'gpu_id': i,
                'caption_type': self.caption_type,
                'caption_tone': self.caption_tone,
                'caption_length': self.caption_length,
                'cachedir': self.workspace.cache.dir('joycaption-alpha-one')
            }
            for i in range(num_gpus)
        ]

        # Create batches of work items
        batches = []
        for i in range(0, len(needs_caption), self.batch_size):
            batch_images = needs_caption[i:i + self.batch_size]
            batch = BatchItem(
                image_indices=list(range(i, min(i + self.batch_size, len(needs_caption)))),
                image_paths=[self.workspace.get_path(img) for img in batch_images]
            )
            batches.append(batch)

        # Create and run the parallel controller
        controller = ParallelController(JoyCaptionAlphaOneWorker, worker_params_list)
        
        with tqdm(total=len(needs_caption), desc="JoyCaption") as pbar:
            for success, result in controller.run(batches):
                if not success:
                    raise RuntimeError(f"Worker failed: {result}")
                
                indices, captions = result
                for idx, caption in zip(indices, captions):
                    needs_caption[idx]._joycaption.value = caption.rstrip('!')
                    self.log.info(f"{self.workspace.get_path(needs_caption[idx])} => {caption}")
                pbar.update(len(indices))
        
        return dataset


__all__ = [ 'JoyCaptionAlphaOne' ]

