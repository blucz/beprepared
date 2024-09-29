from typing import Dict,List
from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.properties import ConstProperty

class AddTags(Node):
    def __init__(self, tags: str | List[str]):
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
    def __init__(self, tags: str | List[str]):
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
    def __init__(self, mapping: Dict[str,str]):
        super().__init__()
        self.mapping = mapping

    def eval(self, dataset: Dataset) -> Dataset:
        for image in dataset.images:
            if image.tags.has_value:
                image.tags.value = set(self.mapping.get(tag, tag) for tag in image.tags.value)
        return dataset
