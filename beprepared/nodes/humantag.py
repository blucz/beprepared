import os
import random

from tqdm import tqdm

from beprepared.node import Node
from beprepared.web import Applet
from beprepared.workspace import Workspace
from beprepared.image import Image
from beprepared.properties import CachedProperty, ConstProperty, ComputedProperty

from fastapi import Request
from fastapi.responses import JSONResponse, FileResponse
from typing import List

from beprepared.web import WebInterface


def get_tags(image):
    if image.human_tags.has_value:
        return image.human_tags.value.get('tags',[])
    return []

def get_rejected(image):
    if image.human_tags.has_value:
        return image.human_tags.value.get('rejected', False)
    return False

class HumanTagApplet(Applet):
    '''HumanTagApplet is a web interface for tagging images. It is used by the HumanTag node to allow users to tag images using a web interface.'''
    def __init__(self, version, tags_with_layout, tags, images_to_tag, cb_tagged=lambda image: None):
        super().__init__('humantag', 'HumanTag')
        self.images_to_tag = images_to_tag
        self.cb_tagged = cb_tagged
        self.version = version
        self.tags_with_layout = tags_with_layout
        self.tags = tags

        # Map image IDs to images and properties
        @self.app.get("/api/images")
        def get_images():
            with_tag    = [image for image in self.images_to_tag if image.human_tags.has_value and image.human_tags.value['version'] == self.version]
            without_tag = [image for image in self.images_to_tag if not image.human_tags.has_value or image.human_tags.value['version'] != self.version]
            images = with_tag + without_tag 
            start_index = len(with_tag)
            images_data = {
                'images': [{"id": idx, "tags": get_tags(image), "rejected": get_rejected(image), "objectid": image.objectid.value} for idx,image in enumerate(images)],
                'start_index': start_index
            }
            return images_data

        @self.app.get("/api/tags")
        def get_images():
            return self.tags_with_layout

        @self.app.get("/objects/{object_id}")
        def get_object(object_id: str):
            path = Workspace.current.get_object_path(object_id)
            return FileResponse(path)

        @self.app.post("/api/images/{image_id}")
        async def update_image(image_id: int, request: Request):
            data = await request.json()
            tags = data.get('tags') 
            if tags is None:
                return JSONResponse({"error": "Invalid tags"}, status_code=400)
            for tag in tags:
                if tag not in self.tags:
                    return JSONResponse({"error": f"Invalid tag: {tag}"}, status_code=400)
            rejected = data.get('rejected', False)
            image = self.images_to_tag[image_id]
            if image is None:
                return JSONResponse({"error": "Invalid image ID"}, status_code=400)
            image.human_tags.value = {'version': self.version, 'tags': tags, 'rejected': rejected}
            if self.cb_tagged(image):
                return {"status": "done"}
            else:
                return {"status": "ok"}

class HumanTag(Node):
    '''HumanTag is a node that allows you to tag images using a web interface.
      
       The domain is used to separate different sets of tags. For example, you could have a domain for tags 
       related to style and a domain for tags related to content. If you want to keep them separate, use 
       different domains. Likewise, if you have multiple sets of images that use different tagging practices, 
       you will want to use different domains.

       If the tag set evolves and you want to evaluate images again without losing work, increment the 
       version number. This will cause the web interface to re-evaluate all images.
    '''
    def __init__(self, 
                 domain: str = 'default', 
                 version = 1, 
                 tags: List[str] | List[List[str]] = [],
                 target_prop: str = 'tags',
                 skip_ui=False):
        '''Initializes the HumanTag node

        Args:
            domain (str): The domain to use for the tags
            version (int): The version of the tags. Increment this to re-evaluate all images
            tags (List[str] or List[List[str]]): The tags to use. If you provide a nested list, e.g. [['tag1', 'tag2'], ['tag3']], the tags will be grouped in the UI
            target_prop (str): The property to store the tags in (default is 'tags')
            skip_ui (bool): If True, the UI will be skipped. This is most useful on large data sets where you want to work on subsequent nodes without waiting for all images to be tagged.
        '''
        super().__init__()
        if len(tags) == 0:
            raise Abort("At least one tag must be provided in HumanTag")
        self.tags = []
        for item in tags:
            if isinstance(item, str):
                self.tags.append(item)
            else:
                self.tags.extend(item)
        self.tags.sort()
        if isinstance(tags[0], list):
            self.tags_with_layout = tags
        else:
            self.tags_with_layout = [tags]
        self.domain = domain
        self.version = version
        self.skip_ui = skip_ui
        self.target_prop = target_prop

    def eval(self, dataset):
        needs_tag = []
        tagged = []

        for image in dataset.images:
            image.human_tags = CachedProperty('humantag', self.domain, image)
            setattr(image, self.target_prop, ComputedProperty(lambda image: image.human_tags.value['tags'] if image.human_tags.has_value else []))

        for image in dataset.images:
            if not image.human_tags.has_value:
                needs_tag.append(image)
            else:
                data = image.human_tags.value
                if data['version'] == self.version:
                    tagged.append(image)
                else:
                    needs_tag.append(image)

        # Print a frequency count table by tag over all the images in tagged:
        tag_freq = {}
        for image in tagged:
            for tag in get_tags(image):
                tag_freq[tag] = tag_freq.get(tag, 0) + 1
        self.log.info("Tag frequency count:")
        for tag in self.tags:
            self.log.info(f"  {tag}: {tag_freq.get(tag, 0)}")

        if len(needs_tag) == 0:
            self.log.info("All images already have been tagged, skipping")
            dataset.images = [image for image in dataset.images if not get_rejected(image)]
            return dataset

        self.log.info(f"Tagging images using human tag for {len(needs_tag)} images")

        if self.skip_ui:
            self.log.info("Skipping UI, some images were not tagged")
            return dataset

        # Run webui with progress bar 
        progress_bar = tqdm(total=len(dataset.images), desc="Tagging images")
        progress_bar.n = len(tagged)
        progress_bar.refresh()
        def image_tagged(image: Image):
            progress_bar.n = len([img for img in dataset.images if img.human_tags.has_value and img.human_tags.value['version'] == self.version])
            progress_bar.refresh()
        applet = HumanTagApplet(self.version, self.tags_with_layout, self.tags, tagged + needs_tag, cb_tagged=image_tagged)
        applet.run(self.workspace)
        progress_bar.close()

        # Apply tag based on results from web interface
        for image in dataset.images:
            data = image.human_tags.value if image.human_tags.has_value else {'version': 0, 'tags': []}
            tags = data['tags']
            existing_tags = getattr(image, self.target_prop).value if getattr(image, self.target_prop).has_value else []
            existing_tags = [tag for tag in existing_tags if tag not in self.tags]
            setattr(image, self.target_prop, ConstProperty(existing_tags + tags))

        dataset.images = [image for image in dataset.images if not get_rejected(image)]

        return dataset

__all__ = [ 'HumanTag' ]
