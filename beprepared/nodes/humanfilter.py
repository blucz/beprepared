import os
import random

from beprepared.node import Node
from beprepared.properties import CachedProperty

from fastapi import Request
from fastapi.responses import JSONResponse, FileResponse

from beprepared.web import WebInterface

from tqdm import tqdm

class HumanFilter(Node):
    def __init__(self, domain: str = 'default'):
        super().__init__()
        self.domain = domain

    def eval(self, dataset):
        needs_filter = []
        already_filtered_count = 0
        for image in dataset.images:
            image.passed_human_filter = CachedProperty('humanfilter', self.domain, image)
            if not image.passed_human_filter.has_value:
                needs_filter.append(image)
            else:
                already_filtered_count += 1

        if len(needs_filter) == 0:
            self.log.info("All images already have been filtered, skipping")
            dataset.images = [image for image in dataset.images if image.passed_human_filter.value]
            return dataset

        def desc():
            accepted_count = len([image for image in dataset.images if image.passed_human_filter.has_value and image.passed_human_filter.value])
            filtered_count = len([image for image in dataset.images if image.passed_human_filter.has_value])
            return f"Human filter ({accepted_count/filtered_count*100:.1f}% accepted)"

        self.log.info(f"Filtering images using human filter for {len(needs_filter)} images (already filtered: {already_filtered_count})")   

        web = WebInterface(name='HumanFilter',
                           static_files_path=os.path.join(os.path.dirname(__file__), 'humanfilter_web', 'static'))

        # Map image IDs to images and properties
        @web.app.get("/api/images")
        def get_images():
            images_data = [{"id": idx, "objectid": image.objectid.value } 
                            for idx,image in enumerate(needs_filter)
                            if not image.passed_human_filter.has_value]
            random.shuffle(images_data)
            return images_data

        @web.app.get("/objects/{object_id}")
        def get_object(object_id: str):
            path = self.workspace.get_object_path(object_id)
            return FileResponse(path)

        @web.app.post("/api/images/{image_id}")
        async def update_image(image_id: int, request: Request):
            data = await request.json()
            action = data.get('action')
            if action not in ['accept', 'reject']:
                return JSONResponse({"error": "Invalid action"}, status_code=400)
            image = needs_filter[image_id]
            if image is None:
                return JSONResponse({"error": "Invalid image ID"}, status_code=400)
            if action == 'reject':
                image.passed_human_filter.value = False
            elif action == 'accept':
                image.passed_human_filter.value = True
            else:
                return JSONResponse({"error": "Invalid action"}, status_code=400)
            progress_bar.n += 1
            progress_bar.set_description(desc())
            progress_bar.refresh()
            return {"status": "ok"}

        progress_bar = tqdm(total=len(dataset.images), desc=desc())
        progress_bar.n = already_filtered_count
        progress_bar.refresh()
        web.run()
        progress_bar.close()

        # Apply filter based on results from web interface
        total_count = len(dataset.images)
        accepted_count = len([image for image in dataset.images if image.passed_human_filter.value])

        self.log.info(f"Human filtering completed, accepted {accepted_count} out of {total_count} images")
        dataset.images = [image for image in dataset.images if image.passed_human_filter.value]

        return dataset
