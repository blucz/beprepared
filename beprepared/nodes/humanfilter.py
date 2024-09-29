import os

from beprepared.node import Node
from beprepared.properties import CachedProperty

from fastapi import Request
from fastapi.responses import JSONResponse, FileResponse

from beprepared.web import WebInterface

class HumanFilter(Node):
    def __init__(self, domain: str = 'default'):
        super().__init__()
        self.domain = domain

    def eval(self, dataset):
        needs_filter = []
        for image in dataset.images:
            image.passed_human_filter = CachedProperty('humanfilter', self.domain, image)
            if not image.passed_human_filter.has_value:
                needs_filter.append(image)

        if len(needs_filter) == 0:
            self.log.info("All images already have been filtered, skipping")
            dataset.images = [image for image in dataset.images if image.passed_human_filter.value]
            return dataset

        self.log.info(f"Filtering images using human filter for {len(needs_filter)} images")

        web = WebInterface(name='HumanFilter',
                           static_files_path=os.path.join(os.path.dirname(__file__), 'humanfilter_web', 'static'))

        # Map image IDs to images and properties
        @web.app.get("/api/images")
        def get_images():
            images_data = [{"id": idx} for idx in range(len(needs_filter))]
            return images_data

        @web.app.get("/images/{image_id}")
        def get_image_file(image_id: int):
            image = needs_filter[image_id]
            if image is None:
                return JSONResponse({"error": "Invalid image ID"}, status_code=400)
            image_path = self.workspace.get_path(image)
            return FileResponse(image_path)

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
            return {"status": "ok"}

        web.run()

        # Apply filter based on results from web interface
        total_count = len(dataset.images)
        accepted_count = len([image for image in dataset.images if image.passed_human_filter.value])

        self.log.info(f"Human filtering completed, accepted {accepted_count} out of {total_count} images")
        dataset.images = [image for image in dataset.images if image.passed_human_filter.value]

        return dataset
