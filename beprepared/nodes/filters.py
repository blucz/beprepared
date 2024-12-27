from beprepared.node import Node
from beprepared.image import Image
from beprepared.dataset import Dataset
from typing import Callable, List


class FilterBySize(Node):
    '''Filters images based on their size. This node can be used to filter images based on their width, height, or edge length'''

    def __init__(self, min_width=None, min_height=None, min_edge=None, max_width=None, max_height=None, max_edge=None):
        '''Initializes the FilterBySize node

        Args:
            min_width (int): The minimum width of the image
            min_height (int): The minimum height of the image
            min_edge (int): The minimum edge length of the image
            max_width (int): The maximum width of the image
            max_height (int): The maximum height of the image
            max_edge (int): The maximum edge length of the image
        '''
        super().__init__()
        self.min_width = min_width
        self.min_height = min_height
        self.min_edge = min_edge
        self.max_width = max_width
        self.max_height = max_height
        self.max_edge = max_edge

    def is_ok(self, image: Image) -> bool:
        if self.min_width and image.width.value < self.min_width: 
            return False
        if self.min_height and image.height.value < self.min_height: 
            return False
        if self.min_edge and min(image.width.value, image.height.value) < self.min_edge: 
            return False
        if self.max_width and image.width.value > self.max_width: 
            return False
        if self.max_height and image.height.value > self.max_height: 
            return False
        if self.max_edge and max(image.width.value, image.height.value) > self.max_edge: 
            return False
        return True

    def eval(self, dataset: Dataset) -> Dataset:
        orig_images = dataset.images
        dataset.images = [image for image in orig_images if self.is_ok(image)]
        self.log.info(f"Kept {len(dataset.images)} out of {len(orig_images)} images ({len(orig_images) - len(dataset.images)} filtered)")
        return dataset

class FilterByAspectRatio(Node):
    '''Filters images based on their aspect ratio (width/height)'''

    def __init__(self, min_aspect=None, max_aspect=None):
        '''Initializes the FilterByAspectRatio node

        Args:
            min_aspect (float): The minimum aspect ratio (width/height) to keep
            max_aspect (float): The maximum aspect ratio (width/height) to keep
        '''
        super().__init__()
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

    def is_ok(self, image: Image) -> bool:
        if self.min_aspect and image.aspect_ratio.value < self.min_aspect:
            return False
        if self.max_aspect and image.aspect_ratio.value > self.max_aspect:
            return False
        return True

    def eval(self, dataset: Dataset) -> Dataset:
        orig_images = dataset.images
        dataset.images = [image for image in orig_images if self.is_ok(image)]
        self.log.info(f"Kept {len(dataset.images)} out of {len(orig_images)} images ({len(orig_images) - len(dataset.images)} filtered)")
        return dataset


class Filter(Node):
    def __init__(self, predicate: Callable[[Image], bool]):
        super().__init__()
        self.predicate = predicate

    def eval(self, dataset: Dataset) -> Dataset:
        orig_images = dataset.images
        dataset.images = [image for image in dataset.images if self.predicate(image)]
        self.log.info(f"Filtered dataset from {len(orig_images)} to {len(dataset.images)} images ({len(orig_images) - len(dataset.images)} filtered)")
        return dataset

