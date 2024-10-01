from beprepared.node import Node
from beprepared.workspace import Workspace
from beprepared.dataset import Dataset

class Dedupe(Node):
    def __init__(self):
        super().__init__()

    def eval(self, dataset: Dataset) -> Dataset:
        visited = set()
        prev_count = len(dataset.images)
        dataset.images = [image for image in dataset.images 
                            if image.objectid.value not in visited and not visited.add(image.objectid.value)]
        self.log.info(f"Removed {prev_count - len(dataset.images)} duplicates from dataset ({100 * (prev_count - len(dataset.images)) / prev_count})%")
        return dataset
