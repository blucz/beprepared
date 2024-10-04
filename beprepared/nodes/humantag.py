import os
import random

from tqdm import tqdm

from beprepared.node import Node
from beprepared.workspace import Workspace
from beprepared.image import Image
from beprepared.properties import CachedProperty, ConstProperty

from fastapi import Request
from fastapi.responses import JSONResponse, FileResponse
from typing import List

from beprepared.web import WebInterface

class HumanTagUi:
    def __init__(self, version, tags_with_layout, tags, images_to_tag, cb_tagged=lambda image: None):
        self.images_to_tag = images_to_tag
        self.cb_tagged = cb_tagged
        self.version = version
        self.tags_with_layout = tags_with_layout
        self.tags = tags

    def run(self):
        self.web = WebInterface(name='HumanTag',
                           static_files_path=os.path.join(os.path.dirname(__file__), 'humantag_web', 'static'),
                           debug=True)

        # Map image IDs to images and properties
        @self.web.app.get("/api/images")
        def get_images():
            with_tag    = [image for image,tags in self.images_to_tag if image.human_tags.has_value and image.human_tags.value['version'] == self.version]
            without_tag = [image for image,tags in self.images_to_tag if not image.human_tags.has_value or image.human_tags.value['version'] != self.version]
            images = with_tag + without_tag 
            start_index = len(with_tag)
            images_data = {
                'images': [{"id": idx, "tags": tags, "objectid": image.objectid.value} for idx,(image,tags) in enumerate(self.images_to_tag)],
                'start_index': start_index
            }
            return images_data

        @self.web.app.get("/api/tags")
        def get_images():
            return self.tags_with_layout

        @self.web.app.get("/objects/{object_id}")
        def get_object(object_id: str):
            path = Workspace.current.get_object_path(object_id)
            return FileResponse(path)

        @self.web.app.post("/api/images/{image_id}")
        async def update_image(image_id: int, request: Request):
            data = await request.json()
            tags = data.get('tags') 
            if tags is None:
                return JSONResponse({"error": "Invalid tags"}, status_code=400)
            for tag in tags:
                if tag not in self.tags:
                    return JSONResponse({"error": f"Invalid tag: {tag}"}, status_code=400)
            image,_ = self.images_to_tag[image_id]
            if image is None:
                return JSONResponse({"error": "Invalid image ID"}, status_code=400)
            image.human_tags.value = {'version': self.version, 'tags': tags}
            if self.cb_tagged(image):
                return {"status": "done"}
            else:
                return {"status": "ok"}

        self.web.run()

    def stop(self):
        if self.web: 
            self.web.stop()

class HumanTag(Node):
    '''Allows a human to manually attach a static set of tags to each image.
      
       Domain is used to separate different sets of tags. For example, you could have a domain for tags 
       related to style and a domain for tags related to content.

       If the tag set evolves and you want to evaluate images again without losing work, increment the 
       version number.
    '''
    def __init__(self, 
                 domain: str = 'default', 
                 version = 1, 
                 tags: List[str] | List[List[str]] = [],
                 target_prop: str = 'tags'):
        super().__init__()
        if len(tags) == 0:
            raise ValueError("At least one tag must be provided")
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

    def eval(self, dataset):
        needs_tag = []
        tagged = []
        for image in dataset.images:
            image.human_tags = CachedProperty('humantag', self.domain, image)
            if not image.human_tags.has_value:
                needs_tag.append((image, []))
            else:
                data = image.human_tags.value
                if data['version'] == self.version:
                    tagged.append((image, data['tags']))
                else:
                    needs_tag.append((image, data['tags']))

        if len(needs_tag) == 0:
            self.log.info("All images already have been tagged, skipping")
            dataset.images = [image for image in dataset.images if image.human_tags.value]
            return dataset

        self.log.info(f"Tagging images using human tag for {len(needs_tag)} images")

        # Run webui with progress bar 
        progress_bar = tqdm(total=len(dataset.images), desc="Tagging images")
        progress_bar.n = len(tagged)
        progress_bar.refresh()
        def image_tagged(image: Image):
            progress_bar.n += 1
            progress_bar.refresh()
        HumanTagUi(self.version, self.tags_with_layout, self.tags, tagged + needs_tag, cb_tagged=image_tagged).run()
        progress_bar.close()

        # Apply tag based on results from web interface
        for image in dataset.images:
            data = image.human_tags.value if image.human_tags.has_value else {'version': 0, 'tags': []}
            tags = data['tags']
            existing_tags = image.tags.value if image.tags.has_value else []
            existing_tags = [tag for tag in existing_tags if tag not in self.tags]
            image.tags = ConstProperty(existing_tags + tags)

        return dataset
