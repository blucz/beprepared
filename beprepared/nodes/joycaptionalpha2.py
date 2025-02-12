import gc
from beprepared.workspace import Workspace
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
from typing import Literal, Union, List
from .utils import tqdm
from .parallelworker import ParallelController, BaseWorker

@dataclass
class BatchItem:
    """Represents a batch of images to process"""
    image_indices: List[int]  # Original indices in the dataset
    image_paths: List[str]    # Paths to the images

CLIP_PATH = "google/siglip-so400m-patch14-384"
CHECKPOINT_PATH = "cgrkzexw-599808"
BASE_MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-Instruct"

CAPTION_TYPE_MAP = {
    "descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "descriptive_informal": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "training_prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "midjourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "booru_tag_list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "booru_like_tag_list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "art_critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "product_listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "social_media_post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
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

class JoyCaptionAlphaTwoWorker(BaseWorker):
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

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)

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
        self.caption_length = self.worker_params['caption_length']
        self.extra_options = self.worker_params['extra_options']
        self.name_input = self.worker_params['name_input']

    def process_item(self, item: BatchItem) -> Tuple[List[int], List[str]]:
        """Process a batch of images and return their captions"""
        input_images = [Image.open(path) for path in item.image_paths]
        
        torch.cuda.empty_cache()
        length = None if self.caption_length == "any" else self.caption_length

        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        if length is None:
            map_idx = 0
        elif isinstance(length, int):
            map_idx = 1
        elif isinstance(length, str):
            map_idx = 2
        else:
            raise ValueError(f"Invalid caption length: {length}")

        prompt_str = CAPTION_TYPE_MAP[self.caption_type][map_idx]

        if len(self.extra_options) > 0:
            prompt_str += " " + " ".join(self.extra_options)

        prompt_str = prompt_str.format(
            name=self.name_input,
            length=self.caption_length,
            word_count=self.caption_length
        )

        # Process all images in batch
        images = [img.resize((384, 384), Image.LANCZOS) for img in input_images]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]
        
        # Convert to tensor and normalize
        pixel_values = torch.stack([
            TVF.normalize(TVF.pil_to_tensor(img).float() / 255.0, [0.5], [0.5])
            for img in images
        ]).to(f'cuda:{self.worker_params["gpu_id"]}')

        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt_str,
            },
        ]

        convo_string = self.tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
        convo_tokens = self.tokenizer.encode(
            convo_string, return_tensors="pt",
            add_special_tokens=False, truncation=False
        ).to(f'cuda:{self.worker_params["gpu_id"]}')
        prompt_tokens = self.tokenizer.encode(
            prompt_str, return_tensors="pt",
            add_special_tokens=False, truncation=False
        ).to(f'cuda:{self.worker_params["gpu_id"]}')
        convo_tokens = convo_tokens.squeeze(0).to(f'cuda:{self.worker_params["gpu_id"]}')
        prompt_tokens = prompt_tokens.squeeze(0).to(f'cuda:{self.worker_params["gpu_id"]}')

        eot_id_indices = (
            convo_tokens == self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ).nonzero(as_tuple=True)[0].tolist()
        preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]

        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = self.clip_model(
                pixel_values=pixel_values, output_hidden_states=True
            )
            embedded_images = self.image_adapter(vision_outputs.hidden_states)
            embedded_images = embedded_images.to(f'cuda:{self.worker_params["gpu_id"]}')

        # Expand conversation embeddings for batch
        convo_embeds = self.text_model.model.embed_tokens(
            convo_tokens.unsqueeze(0).to(f'cuda:{self.worker_params["gpu_id"]}')
        )
        convo_embeds = convo_embeds.expand(len(input_images), -1, -1)

        # Prepare input embeddings for batch
        input_embeds = torch.cat([
            convo_embeds[:, :preamble_len],
            embedded_images.to(dtype=convo_embeds.dtype),
            convo_embeds[:, preamble_len:],
        ], dim=1).to(f'cuda:{self.worker_params["gpu_id"]}')

        # Prepare input IDs for batch
        input_ids = torch.cat([
            convo_tokens[:preamble_len].unsqueeze(0).to(f'cuda:{self.worker_params["gpu_id"]}').expand(len(input_images), -1),
            torch.zeros(
                (len(input_images), embedded_images.shape[1]),
                dtype=torch.long,
                device=f'cuda:{self.worker_params["gpu_id"]}'
            ),
            convo_tokens[preamble_len:].unsqueeze(0).to(f'cuda:{self.worker_params["gpu_id"]}').expand(len(input_images), -1),
        ], dim=1)
        attention_mask = torch.ones_like(input_ids)

        generate_ids = self.text_model.generate(
            input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            suppress_tokens=None
        )

        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == self.tokenizer.eos_token_id or generate_ids[0][-1] == self.tokenizer.convert_tokens_to_ids("<|eot_id|>"):
            generate_ids = generate_ids[:, :-1]

        captions = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Remove any remaining special tokens
        special_tokens = ["<|eot_id|>", "<|finetune_right_pad_id|>"]
        clean_captions = []
        for caption in captions:
            for token in special_tokens:
                caption = caption.replace(token, "")
            clean_captions.append(caption.strip().rstrip('!'))

        return item.image_indices, clean_captions

    def shutdown_worker(self):
        """Clean up GPU resources"""
        del self.clip_model
        del self.tokenizer
        del self.text_model
        del self.image_adapter
        gc.collect()
        torch.cuda.empty_cache()


class JoyCaptionAlphaTwo(Node):
    '''Generates captions for images using the JoyCaption Alpha Two model.'''
    def __init__(self,
                 target_prop: str = 'caption',
                 caption_type: Literal['descriptive', 'descriptive_informal', 'training_prompt', 
                                     'midjourney', 'booru_tag_list', 'booru_like_tag_list', 
                                     'art_critic', 'product_listing', 'social_media_post'] = 'descriptive',
                 caption_length: Union[int, str] = 'long',
                 extra_options: List[str] = [],
                 name_input: str = '',
                 batch_size: int = 4) -> None:
        super().__init__()
        self.target_prop = target_prop
        self.caption_type = caption_type
        self.caption_length = caption_length
        self.extra_options = extra_options
        self.name_input = name_input
        self.batch_size = batch_size

    def eval(self, dataset):
        needs_caption = []
        for image in dataset.images:
            image._joycaption2 = CachedProperty('joycaption2', 'v2', self.caption_type, self.caption_length, 
                                              tuple(self.extra_options), self.name_input, image)
            setattr(image, self.target_prop, ComputedProperty(lambda image: image._joycaption2.value if image._joycaption2.has_value else None))
            if not image._joycaption2.has_value:
                needs_caption.append(image)

        if len(needs_caption) == 0:
            self.log.info(f"All images already have captions, skipping")
            return dataset

        self.log.info(f"Generating captions for {len(needs_caption)} images")

        cachedir = self.workspace.cache.dir('joycaption-alpha-two')
        image_adapter_path = cachedir.download(
            "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/image_adapter.pt",
            "image_adapter.pt"
        )
        clip_model_path = cachedir.download(
            "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/clip_model.pt",
            "clip_model.pt"
        )
        text_model_path = os.path.join(cachedir.dir, 'text_model')
        text_adapter_config_path = cachedir.download(
            "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/text_model/adapter_config.json",
            "text_model/adapter_config.json"
        )
        cachedir.download(
            "https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two/resolve/main/cgrkzexw-599808/text_model/adapter_model.safetensors",
            "text_model/adapter_model.safetensors"
        )

        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        self.log.info(f"Processing {len(needs_caption)} images with JoyCaptionAlphaTwo using {num_gpus} GPUs")
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available")

        # Create worker parameters for each GPU
        worker_params_list = [
            {
                'gpu_id': i,
                'caption_type': self.caption_type,
                'caption_length': self.caption_length,
                'extra_options': self.extra_options,
                'name_input': self.name_input,
                'cachedir': self.workspace.cache.dir('joycaption-alpha-two')
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
        controller = ParallelController(JoyCaptionAlphaTwoWorker, worker_params_list)
        
        with tqdm(total=len(needs_caption), desc="JoyCaptionAlphaTwo") as pbar:
            for success, result in controller.run(batches):
                if not success:
                    raise RuntimeError(f"Worker failed: {result}")
                
                indices, captions = result
                for idx, caption in zip(indices, captions):
                    needs_caption[idx]._joycaption2.value = caption
                    self.log.info(f"{self.workspace.get_path(needs_caption[idx])} => {caption}")
                pbar.update(len(indices))
        
        return dataset

__all__ = ['JoyCaptionAlphaTwo']
