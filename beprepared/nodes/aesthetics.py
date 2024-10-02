import os
import random
from concurrent.futures import ThreadPoolExecutor

from beprepared.node import Node
from beprepared.properties import CachedProperty
from PIL import Image
import torch

from transformers import CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV1

from tqdm import tqdm

class AestheticScore(Node):
    '''Computes an aesthetic score for each image in the dataset using the simple aesthetics predictor model.

    This is most commonly combined with `Sort` and `Filter` nodes in order to select images based on their aesthetic score.'''
    def __init__(self, batch_size=256):
        super().__init__()
        self.batch_size = batch_size

    def eval(self, dataset):
        needs_aesthetic_score = []
        for image in dataset.images:
            image.aesthetic_score = CachedProperty('aesthetic_score', 'v1', image)
            if not image.aesthetic_score.has_value:
                needs_aesthetic_score.append(image)

        if len(needs_aesthetic_score) == 0:
            self.log.info("All images already have aesthetic scores, skipping")
            return dataset

        model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"

        device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
        num_gpus = torch.cuda.device_count()
        
        model = AestheticsPredictorV1.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)

        if num_gpus > 1:
            model = torch.nn.DataParallel(model)

        def infer_batch(images):
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                return model(**inputs)

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            for i in tqdm(range(0, len(needs_aesthetic_score), self.batch_size), desc="Computing Aesthetic Scores"):
                batch_images = needs_aesthetic_score[i:i + self.batch_size]
                batch_pilimages = list(executor.map(lambda img: Image.open(self.workspace.get_path(img)), batch_images))
                batch_logits = infer_batch(batch_pilimages).logits.tolist()
                for image, logits in zip(batch_images, batch_logits):
                    image.aesthetic_score.value = logits[0]

        return dataset
