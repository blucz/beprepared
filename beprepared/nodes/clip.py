import os
import random
from concurrent.futures import ThreadPoolExecutor

from beprepared.node import Node
from beprepared.properties import CachedProperty
from PIL import Image
import torch

from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

class ClipModel:
    DEFAULT                    = 'openai/clip-vit-large-patch14'
    CLIP_VIT_LARGE_PATCH14     = 'openai/clip-vit-large-patch14'
    CLIP_VIT_LARGE_PATCH14_336 = 'openai/clip-vit-large-patch14-336'
    CLIP_VIT_BASE_PATCH16      = 'openai/clip-vit-base-patch16'
    CLIP_VIT_BASE_PATCH32      = 'openai/clip-vit-base-patch32'

class ClipEmbed(Node):
    DEFAULT_MODEL = 'openai/clip-vit-large-patch14'
    
    def __init__(self, model: str = ClipModel.DEFAULT, batch_size=128, target_prop='clip'):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.target_prop = target_prop

    def eval(self, dataset):
        needs_encoding = []
        for image in dataset.images:
            prop = CachedProperty('clip_embedding', self.model, image)
            setattr(image, self.target_prop, prop)
            if not prop.has_value:
                needs_encoding.append(image)

        if len(needs_encoding) == 0:
            self.log.info("All images already have been embedded, skipping")
            return dataset


        device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
        num_gpus = torch.cuda.device_count()
        
        clip_model = CLIPModel.from_pretrained(self.model).to(device)
        clip_processor = CLIPProcessor.from_pretrained(self.model)
        
        if num_gpus > 1:
            clip_model = torch.nn.DataParallel(clip_model)
            get_image_features = clip_model.module.get_image_features
        else:
            get_image_features = clip_model.get_image_features

        try:
            clip_model = torch.compile(clip_model)
        except Exception as e:
            self.log.warning(f"torch.compile failed: {e}. Proceeding without compilation.")
        
        # Function to compute CLIP embeddings for a batch of images
        def get_clip_embeddings_batch(images):
            inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_features = get_image_features(**inputs)
            return image_features.cpu().numpy()

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for i in tqdm(range(0, len(needs_encoding), self.batch_size), desc="CLIP Embedding"):
                batch_images = needs_encoding[i:i + self.batch_size]
                pil_images = list(executor.map(lambda img: Image.open(self.workspace.get_path(img)), batch_images))
                embeddings = get_clip_embeddings_batch(pil_images)
                for image, embedding in zip(batch_images, embeddings):
                    prop = getattr(image, self.target_prop)
                    prop.value = embedding.tolist()

        return dataset

