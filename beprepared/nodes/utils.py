from beprepared.properties import ConstProperty
from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.image import Image

from typing import List, Any, Callable
import random
import shutil
import textwrap 

class Concat(Node):
    def __init__(self):
        super().__init__()

    def eval(self, datasets: List[Dataset]) -> Dataset:
        dataset = Dataset()
        visited = set()
        for input_dataset in datasets:
            for i in input_dataset.images:
                if i in visited: 
                    continue
                visited.add(i)
                dataset.images.append(i)
        self.log.info(f"Concatenated {len(datasets)} datasets into one with {len(dataset.images)} images")
        return dataset

class Info(Node):
    def __init__(self, include_hidden=False):
        super().__init__()
        self.include_hidden = include_hidden

    def eval(self, dataset) -> Dataset:
        if len(dataset.images) == 0:
            return
        max_length_of_propname = max(len(k) for image in dataset.images for k in image.props.keys())
        terminal_cols = shutil.get_terminal_size().columns 
        for image in dataset.images:
            last_was_multiline = False
            print("-" * terminal_cols)
            for k,v in image.props.items():
                if k.startswith('_') and not self.include_hidden:
                    continue
                if not v.has_value: continue
                value = v.value
                if hasattr(value, 'show'):
                    s = value.show()
                elif isinstance(value, list) and len(value) > 50:
                    tostr = lambda x: f'{x:.3f}' if isinstance(x, float) else str(x)
                    s = f"[{', '.join(tostr(x) for x in (value[:25] + ['...'] + value[-25:]))}]"
                else:
                    s = repr(value)
                    propval_cols = max(40, terminal_cols - max_length_of_propname - 4)
                    if len(s) > propval_cols:
                        s = textwrap.fill(s, propval_cols)
                if '\n' in s:
                    pad = ' ' * (2 + max_length_of_propname)
                    s = s.replace('\n', f'\n{pad}')
                if last_was_multiline or '\n' in s:
                    print("")
                print(f"{k.ljust(max_length_of_propname)}  {s}")
                last_was_multiline = '\n' in s
        if len(dataset.images) > 0:
            print("-" * terminal_cols)
        return dataset

class SetCaption(Node):
    def __init__(self, caption):
        super().__init__()
        self.caption = caption

    def eval(self, dataset) -> Dataset:
        for image in dataset.images:
            caption_prop = ConstProperty(self.caption)
            image.caption = caption_prop
        return dataset

class Take(Node):
    def __init__(self, n: int, random: bool=False):
        super().__init__()
        self.n = n
        self.random = random

    def eval(self, dataset):
        if self.random:
            dataset.images = random.sample(dataset.images, self.n)
        else:
            dataset.images = dataset.images[:self.n]
        return dataset

class Filter(Node):
    def __init__(self, predicate: Callable[Image, bool]):
        super().__init__()
        self.predicate = predicate  

    def eval(self, dataset):
        dataset.images = [image for image in dataset.images if self.predicate(image)]
        return dataset


class Sorted(Node):
    def __init__(self, key: Callable[Image, Any], reverse: bool=False):
        super().__init__()
        self.key = key
        self.reverse = reverse

    def eval(self, dataset):
        dataset.images.sort(key=self.key, reverse=self.reverse)
        return dataset


class Shuffle(Node):
    def __init__(self):
        super().__init__()

    def eval(self, dataset):
        random.shuffle(dataset.images)
        return dataset

class Map(Node):
    def __init__(self, func: Callable[Image, Image]):
        super().__init__()
        self.func = func

    def eval(self, dataset):
        dataset.images = [self.func(image) for image in dataset.images]
        return dataset

class Apply(Node):
    def __init__(self, func: Callable[Image, None]):
        super().__init__()
        self.func = func

    def eval(self, dataset):
        for i in dataset.images:
            self.func(i)
        return dataset

class Set(Node):
    def __init__(self, **kwargs):
        super().__init__()
        self.props = kwargs

    def eval(self, dataset):
        for image in dataset.images:
            for k,v in self.props.items():
                if callable(v):
                    setattr(image, k, v(image))
                else:
                    setattr(image, k, ConstProperty(v))
        return dataset
