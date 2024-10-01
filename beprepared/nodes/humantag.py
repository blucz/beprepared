import os

from beprepared.node import Node
from beprepared.properties import CachedProperty, ConstProperty

from fastapi import Request
from fastapi.responses import JSONResponse, FileResponse
from typing import List

from beprepared.web import WebInterface

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
        for image in dataset.images:
            image.human_tags = CachedProperty('humantag', self.domain, image)
            if not image.human_tags.has_value:
                needs_tag.append((image, []))
            else:
                data = image.human_tags.value
                if data['version'] != self.version:
                    needs_tag.append((image, data['tags']))

        if len(needs_tag) == 0:
            self.log.info("All images already have been taged, skipping")
            dataset.images = [image for image in dataset.images if image.human_tags.value]
            return dataset

        self.log.info(f"Tagging images using human tag for {len(needs_tag)} images")

        web = WebInterface(name='HumanTag',
                           static_files_path=os.path.join(os.path.dirname(__file__), 'humantag_web', 'static'))

        # Map image IDs to images and properties
        @web.app.get("/api/images")
        def get_images():
            images_data = [{"id": idx, "tags": tags, "objectid": image.objectid.value} for idx,(version,tags) in enumerate(needs_tag)]
            return images_data

        @web.app.get("/api/tags")
        def get_images():
            return self.tags_with_layout

        @web.app.get("/objects/{object_id}")
        def get_object(object_id: str):
            path = self.workspace.get_object_path(object_id)
            print("PATH", path)
            return FileResponse(path)

        @web.app.post("/api/images/{image_id}")
        async def update_image(image_id: int, request: Request):
            data = await request.json()
            tags = data.get('tags') 
            if tags is None:
                return JSONResponse({"error": "Invalid tags"}, status_code=400)
            for tag in tags:
                if tag not in self.tags:
                    return JSONResponse({"error": f"Invalid tag: {tag}"}, status_code=400)
            image,_ = needs_tag[image_id]
            if image is None:
                return JSONResponse({"error": "Invalid image ID"}, status_code=400)
            image.human_tags.value = {'version': self.version, 'tags': tags}
            return {"status": "ok"}

        web.run()

        # Apply tag based on results from web interface
        for image in dataset.images:
            data = image.human_tags.value if image.human_tags.has_value else {'version': 0, 'tags': []}
            tags = data['tags']
            existing_tags = image.tags.value if image.tags.has_value else []
            existing_tags = [tag for tag in existing_tags if tag not in self.tags]
            image.tags = ConstProperty(existing_tags + tags)

        return dataset
