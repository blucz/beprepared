import os
import gc
from typing import Optional, Literal, List, Tuple
from dataclasses import dataclass

from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria, LogitsProcessor
import torch
from .utils import tqdm
from PIL import Image

from beprepared.workspace import Workspace
from beprepared.node import Node 
from beprepared.properties import CachedProperty, ComputedProperty
from beprepared.nodes.convert_format import convert_image
from .parallelworker import ParallelController, BaseWorker

@dataclass
class BatchItem:
    """Represents a batch of images to process"""
    image_indices: List[int]  # Original indices in the dataset
    image_paths: List[str]    # Paths to the images

# define the prompt template
def apply_prompt_template(prompt):
    s = (
            '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
            "The assistant gives helpful and detailed answers to the user's questions.<|end|>\n"
            f'<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n'
        )
    return s 

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [32007]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids      

class EosLogitProcessor(LogitsProcessor):
    def __init__(self, eos_token_id: int, end_token_id: int):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.end_token_id = end_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) > 1: # Expect at least 1 output token.
            forced_eos = torch.full((scores.size(1),), -float("inf"), device=input_ids.device)
            forced_eos[self.eos_token_id] = 0

            # Force generation of EOS after the <|end|> token.
            scores[input_ids[:, -1] == self.end_token_id] = forced_eos
        return scores 


class XGenMMCaptionWorker(BaseWorker):
    def initialize_worker(self):
        """Initialize the XGen-MM model and processors"""
        gpu_id = self.worker_params['gpu_id']
        torch.cuda.set_device(gpu_id)
        
        MODEL = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(f'cuda:{gpu_id}')
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False, legacy=False)
        self.image_processor = AutoImageProcessor.from_pretrained(MODEL, trust_remote_code=True)
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)

        # Process prompt once
        self.prompt = self.worker_params['prompt']
        query = apply_prompt_template(self.prompt)
        self.language_inputs = self.tokenizer(
            query, 
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length, 
            truncation=True
        )
        self.language_inputs = {name: tensor.to(f'cuda:{gpu_id}') for name, tensor in self.language_inputs.items()}

    def process_item(self, item: BatchItem) -> Tuple[List[int], List[str]]:
        """Process a batch of images and return their captions"""
        cropped_images = []
        image_sizes = []

        for path in item.image_paths:
            raw_image = Image.open(path).convert('RGB')
            w,h = raw_image.size
            cropped_image = raw_image.crop((0, h * 0.05, w, h * 0.91))    # l,t,r,b
            cropped_images.append(cropped_image)
            image_sizes.append(cropped_image.size)

        inputs = [self.image_processor([img], return_tensors="pt", image_aspect_ratio='anyres') for img in cropped_images]
        batch_inputs = {}
        for sample_input in inputs:
            for name, tensor in sample_input.items():
                if name in batch_inputs:
                    batch_inputs[name].append(tensor.squeeze(0).to(f'cuda:{self.worker_params["gpu_id"]}'))
                else:
                    batch_inputs[name] = [tensor.squeeze(0).to(f'cuda:{self.worker_params["gpu_id"]}')]

        # repeat the language inputs for each image
        batch_language_inputs = {}
        for key in self.language_inputs:
            batch_language_inputs[key] = self.language_inputs[key].repeat(len(cropped_images), 1)
        batch_inputs.update(batch_language_inputs)
        batch_inputs['pixel_values'] = [x.to(torch.float16).to(f'cuda:{self.worker_params["gpu_id"]}') for x in batch_inputs['pixel_values']]

        generated_text = self.model.generate(
            **batch_inputs, 
            image_size=image_sizes,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False, 
            max_new_tokens=4096, 
            top_p=None, 
            num_beams=1,
            logits_processor=[EosLogitProcessor(
                eos_token_id=self.tokenizer.eos_token_id, 
                end_token_id=32007
            )],
        )
        predictions = self.tokenizer.batch_decode(generated_text, skip_special_tokens=True)
        captions = []
        for prediction in predictions:
            prediction = prediction.split('<|end|>')[0]
            captions.append(prediction.strip())

        return item.image_indices, captions

    def shutdown_worker(self):
        """Clean up GPU resources"""
        del self.model
        del self.tokenizer
        del self.image_processor
        gc.collect()
        torch.cuda.empty_cache()


class XGenMMCaption(Node):
    '''Generates image captions using the xGen-mm model (aka BLIP3)'''
    DEFAULT_PROMPT = """Your task is to write a caption for this image. Be sure to clearly describe any subjects of the image and their characteristics and position within the image. Also describe aspects of the background, lighting, style, and quality if those details are apparent. """.strip()

    def __init__(self,
                 target_prop:    str                              = 'caption',
                 prompt:         Optional[str] = None,
                 instructions:   Optional[str] = None,
                 batch_size:     int           = 4):
        '''Initializes the XGenMMCaption node

        Args:
            target_prop (str): The property to store the caption in (default is 'caption')
            prompt (str): The prompt to use for the xGen-mm model (read the code)
            instructions (str): Additional instructions to include in the prompt
            batch_size (int): The number of images to process in parallel. If you are running out of memory, try reducing this value.
        '''
        super().__init__()
        self.target_prop  = target_prop
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.batch_size = batch_size
        if instructions:
            self.prompt = f"{self.prompt}\n\n{instructions}"

    def eval(self, dataset):
        needs_caption = []
        for image in dataset.images:
            image._xgenmm_caption = CachedProperty('xgen-mm', 'v1', self.prompt, image)
            setattr(image, self.target_prop, ComputedProperty(lambda image: image._xgenmm_caption.value if image._xgenmm_caption.has_value else None))
            if not image._xgenmm_caption.has_value:
                needs_caption.append(image)

        if len(needs_caption) == 0: 
            self.log.info("All images already have captions, skipping")
            return dataset

        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        self.log.info(f"Processing {len(needs_caption)} images with xGen-mm using {num_gpus} GPUs")
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available")

        # Create worker parameters for each GPU
        worker_params_list = [
            {'gpu_id': i, 'prompt': self.prompt}
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
        controller = ParallelController(XGenMMCaptionWorker, worker_params_list)
        
        with tqdm(total=len(needs_caption), desc="xGen-mm") as pbar:
            for success, result in controller.run(batches):
                if not success:
                    raise RuntimeError(f"Worker failed: {result}")
                
                indices, captions = result
                for idx, caption in zip(indices, captions):
                    needs_caption[idx]._xgenmm_caption.value = caption
                    self.log.info(f"Generated caption for {needs_caption[idx].objectid.value}: {caption}")
                pbar.update(len(indices))
        
        return dataset
