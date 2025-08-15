"""Image cropping nodes for aspect ratio normalization."""

from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.image import Image
from beprepared.properties import CachedProperty, ConstProperty
from beprepared.utils import shorten_path
from beprepared.nodes.utils import tqdm
from PIL import Image as PILImage
from io import BytesIO
from typing import List, Tuple
import json


class CropToAspect(Node):
    """Crops images to the closest matching aspect ratio from a provided list.
    
    This node crops images to match one of the specified aspect ratios, choosing
    the ratio closest to the original image's aspect ratio. It uses center cropping
    to maintain the most important content in the middle of the image.
    
    Examples:
        # Crop to square, portrait, or landscape
        CropToAspect(aspect_ratios=[1.0, 0.75, 1.33])
        
        # Crop to common social media aspect ratios
        CropToAspect(aspect_ratios=[1.0, 1.91, 0.8])
    """
    
    def __init__(self, aspect_ratios: List[float]):
        """Initialize the CropToAspect node.
        
        Args:
            aspect_ratios: List of target aspect ratios (width/height).
                          E.g., [0.5, 1.0, 1.5] for portrait, square, landscape
        """
        super().__init__()
        if not aspect_ratios:
            raise ValueError("aspect_ratios must contain at least one ratio")
        
        self.aspect_ratios = sorted(aspect_ratios)
        
    def find_closest_aspect_ratio(self, original_ratio: float) -> float:
        """Find the closest aspect ratio from the list to the original ratio.
        
        Args:
            original_ratio: The original image's aspect ratio (width/height)
            
        Returns:
            The closest matching aspect ratio from the configured list
        """
        min_diff = float('inf')
        closest = self.aspect_ratios[0]
        
        for ratio in self.aspect_ratios:
            diff = abs(original_ratio - ratio)
            if diff < min_diff:
                min_diff = diff
                closest = ratio
                
        return closest
    
    def calculate_crop_dimensions(self, width: int, height: int, target_ratio: float) -> Tuple[int, int, int, int]:
        """Calculate the crop box for center cropping to target aspect ratio.
        
        Args:
            width: Original image width
            height: Original image height
            target_ratio: Target aspect ratio (width/height)
            
        Returns:
            Tuple of (left, top, right, bottom) crop coordinates
        """
        original_ratio = width / height
        
        if target_ratio > original_ratio:
            # Target is wider than original - crop height
            new_height = int(width / target_ratio)
            new_width = width
        else:
            # Target is taller than original - crop width
            new_width = int(height * target_ratio)
            new_height = height
            
        # Calculate center crop coordinates
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        
        return (left, top, right, bottom)
    
    def eval(self, dataset: Dataset) -> Dataset:
        """Process the dataset, cropping images to matching aspect ratios.
        
        Args:
            dataset: Input dataset containing images to crop
            
        Returns:
            Dataset with cropped images matching target aspect ratios
        """
        output_dataset = Dataset()
        
        # Create cache key for this node's configuration
        config_key = json.dumps({
            'aspect_ratios': self.aspect_ratios
        }, sort_keys=True)
        
        for image in tqdm(dataset.images, desc="Cropping to aspect ratios"):
            # Check if we've already processed this image with these settings
            crop_prop = CachedProperty('croptoaspect', image, config_key)
            
            if crop_prop.has_value:
                # Use cached result
                crop_data = crop_prop.value
                
                # Restore cached crop information
                cropped_objectid = ConstProperty(crop_data['objectid'])
                cropped_width = ConstProperty(crop_data['width'])
                cropped_height = ConstProperty(crop_data['height'])
                target_ratio = ConstProperty(crop_data['target_ratio'])
                
            else:
                # Calculate crop for this image
                original_ratio = image.width.value / image.height.value
                target_ratio_value = self.find_closest_aspect_ratio(original_ratio)
                
                crop_box = self.calculate_crop_dimensions(
                    image.width.value,
                    image.height.value,
                    target_ratio_value
                )
                
                # Load and crop the image
                ws = self.workspace
                image_bytes = ws.db.get_object(image.objectid.value)
                pil_image = PILImage.open(BytesIO(image_bytes))
                
                # Perform the crop
                cropped_pil = pil_image.crop(crop_box)
                
                # Convert to RGB if necessary (for consistency)
                if cropped_pil.mode not in ('RGB', 'RGBA'):
                    cropped_pil = cropped_pil.convert('RGB')
                
                # Save cropped image to database
                output_buffer = BytesIO()
                save_format = image.format.value if image.format.value != 'WEBP' else 'PNG'
                cropped_pil.save(output_buffer, format=save_format, quality=95)
                cropped_bytes = output_buffer.getvalue()
                
                cropped_objectid_value = ws.db.put_object(cropped_bytes)
                
                # Cache the crop information
                crop_data = {
                    'objectid': cropped_objectid_value,
                    'width': cropped_pil.width,
                    'height': cropped_pil.height,
                    'target_ratio': target_ratio_value
                }
                crop_prop.value = crop_data
                
                # Create properties for the cropped image
                cropped_objectid = ConstProperty(cropped_objectid_value)
                cropped_width = ConstProperty(cropped_pil.width)
                cropped_height = ConstProperty(cropped_pil.height)
                target_ratio = ConstProperty(target_ratio_value)
                
                self.log.debug(f"Cropped {shorten_path(image.original_path.value)}: "
                             f"{image.width.value}x{image.height.value} -> "
                             f"{cropped_pil.width}x{cropped_pil.height} "
                             f"(ratio {target_ratio_value:.2f})")
            
            # Create new image object with cropped properties
            cropped_image = image.with_props({
                'objectid': cropped_objectid,
                'width': cropped_width,
                'height': cropped_height,
                'crop_target_ratio': target_ratio,
                'crop_original_width': image.width,
                'crop_original_height': image.height,
            })
            
            output_dataset.images.append(cropped_image)
        
        self.log.info(f"Cropped {len(output_dataset.images)} images to aspect ratios {self.aspect_ratios}")
        
        return output_dataset