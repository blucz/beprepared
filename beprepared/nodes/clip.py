import os
import random
from concurrent.futures import ThreadPoolExecutor

from beprepared.node import Node
from beprepared.properties import CachedProperty
from PIL import Image
import torch

from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

class ClipEmbed(Node):
    '''Computes CLIP embeddings for each image in the dataset using the CLIP model.

    Many of the other nodes depend on CLIP embeddings to work properly, and will error out if CLIP embeddings are not computed.

    CLIP embedding is quite fast on most GPUs and well worth it, as it unlocks a lot of functionality.'''
    MODEL = 'openai/clip-vit-large-patch14'
    
    def __init__(self, batch_size=128, target_prop='clip'):
        '''Create a new ClipEmbed node.

        Args:
            batch_size (int): The number of images to process in parallel. Larger batch sizes will use more memory but may be faster. Default is 128. If you are running out of memory, try reducing this value.
        '''
        super().__init__()
        self.batch_size = batch_size
        self.target_prop = target_prop

    def eval(self, dataset):
        needs_encoding = []
        for image in dataset.images:
            prop = CachedProperty('clip_embedding', ClipEmbed.MODEL, image)
            setattr(image, self.target_prop, prop)
            if not prop.has_value:
                needs_encoding.append(image)

        if len(needs_encoding) == 0:
            self.log.info("All images already have been embedded, skipping")
            return dataset


        device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
        num_gpus = torch.cuda.device_count()
        
        clip_model = CLIPModel.from_pretrained(ClipEmbed.MODEL).to(device)
        clip_processor = CLIPProcessor.from_pretrained(ClipEmbed.MODEL)
        
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

