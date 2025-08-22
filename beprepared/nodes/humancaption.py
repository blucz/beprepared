from .utils import tqdm

from beprepared.node import Node
from beprepared.web import Applet
from beprepared.workspace import Workspace
from beprepared.image import Image
from beprepared.properties import CachedProperty, ConstProperty, ComputedProperty

from fastapi import Request
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
import traceback

from beprepared.web import WebInterface

class HumanCaptionApplet(Applet):
    '''HumanCaptionApplet is a web interface for editing image captions. It is used by the HumanCaption node to allow users to edit captions using a web interface.'''
    def __init__(self, version, images_to_caption, source_prop, cb_captioned=lambda image: None):
        super().__init__('humancaption', 'HumanCaption')
        self.images_to_caption = images_to_caption
        self.cb_captioned = cb_captioned
        self.version = version
        self.source_prop = source_prop

        # Map image IDs to images and properties
        @self.app.get("/api/images")
        def get_images():
            try:
                # Separate images that have been edited vs not
                with_caption = []
                without_caption = []
                
                for image in self.images_to_caption:
                    if image.human_caption.has_value:
                        data = image.human_caption.value
                        if data.get('version') == self.version:
                            with_caption.append(image)
                        else:
                            without_caption.append(image)
                    else:
                        without_caption.append(image)
                
                images = with_caption + without_caption 
                start_index = len(with_caption)
                
                # Build response with captions
                images_list = []
                for idx, image in enumerate(images):
                    # Get the caption - check edited first, then source
                    caption = ""
                    if image.human_caption.has_value:
                        data = image.human_caption.value
                        if isinstance(data, dict):
                            caption = data.get('caption', '')
                    
                    # If no edited caption, get from source property
                    if not caption:
                        source = getattr(image, self.source_prop)
                        if source.has_value:
                            caption = source.value
                    
                    images_list.append({
                        "id": idx, 
                        "caption": str(caption) if caption else "",
                        "objectid": image.objectid.value if image.objectid.has_value else ""
                    })
                
                images_data = {
                    'images': images_list,
                    'start_index': start_index
                }
                return images_data
            except Exception as e:
                import traceback
                print(f"Error in /api/images: {e}")
                print(traceback.format_exc())
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/objects/{object_id}")
        def get_object(object_id: str):
            path = Workspace.current.get_object_path(object_id)
            return FileResponse(path)

        @self.app.post("/api/images/{image_id}")
        async def update_image(image_id: int, request: Request):
            data = await request.json()
            caption = data.get('caption')
            if caption is None:
                return JSONResponse({"error": "Invalid caption"}, status_code=400)
            image = self.images_to_caption[image_id]
            if image is None:
                return JSONResponse({"error": "Invalid image ID"}, status_code=400)
            image.human_caption.value = {'version': self.version, 'caption': caption}
            if self.cb_captioned(image):
                return {"status": "done"}
            else:
                return {"status": "ok"}

class HumanCaption(Node):
    '''HumanCaption is a node that allows you to edit image captions using a web interface.
      
       The source_prop parameter specifies which property to read the initial caption from (e.g., 'caption' from auto-captioning nodes).
       The target_prop parameter specifies where to store the edited caption.
       
       The domain is used to separate different sets of captions. For example, you could have a domain for short 
       captions and a domain for detailed captions. If you want to keep them separate, use different domains.

       If you want to re-edit captions without losing work, increment the version number. This will cause the 
       web interface to re-evaluate all images.
    '''
    def _extract_caption_value(self, image, prop_name):
        """Extract the actual string value from a property"""
        # PropertyBag always returns a Property, never None
        prop = getattr(image, prop_name)
        if prop.has_value:
            return prop.value
        return ""
    
    def __init__(self, 
                 source_prop: str = 'caption',
                 target_prop: str = 'caption',
                 domain: str = 'default', 
                 version: int = 1,
                 skip_ui: bool = False):
        '''Initializes the HumanCaption node

        Args:
            source_prop (str): The property to read the initial caption from (default is 'caption')
            target_prop (str): The property to store the edited caption in (default is 'caption')
            domain (str): The domain to use for the captions
            version (int): The version of the captions. Increment this to re-evaluate all images
            skip_ui (bool): If True, the UI will be skipped. This is most useful on large data sets where you want to work on subsequent nodes without waiting for all images to be captioned.
        '''
        super().__init__()
        self.source_prop = source_prop
        self.target_prop = target_prop
        self.domain = domain
        self.version = version
        self.skip_ui = skip_ui

    def eval(self, dataset):
        needs_caption = []
        captioned = []

        # Set up ComputedProperty for target_prop that returns edited caption or source caption
        for image in dataset.images:
            image.human_caption = CachedProperty('humancaption', image, domain=self.domain)
            
            # Capture the original property value BEFORE overwriting to avoid recursion
            original_prop = getattr(image, self.source_prop)
            original_value = original_prop.value if original_prop.has_value else ""
            
            # Set target_prop as a ComputedProperty that returns the appropriate caption
            def make_caption_getter(orig_val):
                def get_caption(img):
                    # First check if there's an edited caption
                    if img.human_caption.has_value:
                        data = img.human_caption.value
                        if isinstance(data, dict):
                            caption = data.get('caption', '')
                            if caption:
                                return caption
                    # Fall back to the captured original value
                    return orig_val
                return get_caption
            
            setattr(image, self.target_prop, ComputedProperty(make_caption_getter(original_value)))

        for image in dataset.images:
            if not image.human_caption.has_value:
                needs_caption.append(image)
            else:
                data = image.human_caption.value
                if data['version'] == self.version:
                    captioned.append(image)
                else:
                    needs_caption.append(image)

        if len(needs_caption) == 0:
            self.log.info("All images already have been captioned, skipping")
            return dataset

        self.log.info(f"Editing captions for {len(needs_caption)} images")

        if self.skip_ui:
            self.log.info("Skipping UI, some images were not captioned")
            return dataset

        # Run webui with progress bar 
        progress_bar = tqdm(total=len(dataset.images), desc="Editing captions")
        progress_bar.n = len(captioned)
        progress_bar.refresh()
        def image_captioned(image: Image):
            progress_bar.n = len([img for img in dataset.images if img.human_caption.has_value and img.human_caption.value['version'] == self.version])
            progress_bar.refresh()
        applet = HumanCaptionApplet(self.version, captioned + needs_caption, self.source_prop, cb_captioned=image_captioned)
        applet.run(self.workspace)
        progress_bar.close()

        # No need to set captions at the end - ComputedProperty handles it automatically
        return dataset

__all__ = [ 'HumanCaption' ]
