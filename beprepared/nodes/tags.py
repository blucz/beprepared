from typing import Dict,List
from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.properties import ConstProperty

class AddTags(Node):
    '''Adds tags to all images in a dataset'''
    def __init__(self, tags: str | List[str]):
        '''Initializes the AddTags node

        Args:
            tags (str or List[str]): The tags to add to the images
        '''
        super().__init__()
        if isinstance(tags, str):
            tags = [tags]
        self.tags = tags

    def eval(self, dataset) -> Dataset:
        for image in dataset.images:
            if not image.tags.has_value:
                image.tags = ConstProperty(set(self.tags))
                continue
            else:
                image.tags = ConstProperty(image.tags.value.union(self.tags))
        return dataset

class RemoveTags(Node):
    '''Removes tags from all images in a dataset'''
    def __init__(self, tags: str | List[str]):
        '''Initializes the RemoveTags node

        Args:
            tags (str or List[str]): The tags to remove from the images
        '''
        super().__init__()
        if isinstance(tags, str):
            tags = [tags]
        self.tags = tags

    def eval(self, dataset) -> Dataset:
        for image in dataset.images:
            if image.tags.has_value:
                image.tags.value = image.tags.value.difference(self.tags)
                continue
            else:
                image.tags.value.update(self.tags)
        return dataset

class RewriteTags(Node):
    '''Rewrites tags in all images in a dataset'''
    def __init__(self, mapping: Dict[str,str]):
        '''Initializes the RewriteTags node

        Args:
            mapping (Dict[str,str]): A dictionary mapping old tags to new tags
        '''
        super().__init__()
        self.mapping = mapping

    def eval(self, dataset: Dataset) -> Dataset:
        for image in dataset.images:
            if image.tags.has_value:
                image.tags.value = set(self.mapping.get(tag, tag) for tag in image.tags.value)
        return dataset
