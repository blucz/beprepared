from beprepared.properties import ConstProperty
from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.image import Image

from typing import List, Any, Callable
import random
import shutil
import textwrap 

class Concat(Node):
    '''Concatenates multiple datasets into one.

    Because Python operator overloading is limited, usage is as follows:

    set_a = Load("dir_a")
    set_b = Load("dir_b")
    set_c = Load("dir_c")

    (Concat() << set_a << set_b << set_c) >> Save("output_dir")
    '''
    def __init__(self):
        '''Initializes the Concat node'''
        super().__init__()
        for node in nodes:
            node >> self

    def eval(self, datasets) -> Dataset:
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
    '''Prints information about images in a dataset to stdout. This can be useful for debugging small datasets. 

    For larger datasets, the `Save` node generates an `index.html` file in the output directory, and this provides a better experience. '''
    def __init__(self, include_hidden_properties=False):
        '''Initializes the Info node

        Args:
            include_hidden_properties (bool): Whether to include hidden properties (default is False)
        '''
        super().__init__()
        self.include_hidden_properties = include_hidden_properties

    def eval(self, dataset) -> Dataset:
        if len(dataset.images) == 0:
            return
        max_length_of_propname = max(len(k) for image in dataset.images for k in image.props.keys())
        terminal_cols = shutil.get_terminal_size().columns 
        for image in dataset.images:
            last_was_multiline = False
            print("-" * terminal_cols)
            for k,v in image.props.items():
                if k.startswith('_') and not self.include_hidden_properties:
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
    '''Sets a caption for all images in a dataset'''
    def __init__(self, caption: str):
        '''Initializes the SetCaption node

        Args:
            caption (str): The caption to set for all images
        '''
        super().__init__()
        self.caption = caption

    def eval(self, dataset) -> Dataset:
        for image in dataset.images:
            caption_prop = ConstProperty(self.caption)
            image.caption = caption_prop
        return dataset

class Take(Node):
    '''Takes the first `n` images from a dataset'''
    def __init__(self, n: int, random: bool=False):
        '''Initializes the Take node

        Args:
            n (int): The number of images to take
            random (bool): Whether to take the images randomly (default is False)
        '''
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
    '''Filters images in a dataset based on a predicate'''
    def __init__(self, predicate: Callable[Image, bool]):
        '''Initializes the Filter node

        Args:
            predicate (Callable[[Image], bool]): The predicate to filter images with
        '''
        super().__init__()
        self.predicate = predicate  

    def eval(self, dataset):
        dataset.images = [image for image in dataset.images if self.predicate(image)]
        return dataset


class Sorted(Node):
    '''Sorts images in a dataset based on a key'''
    def __init__(self, key: Callable[Image, Any], reverse: bool=False):
        '''Initializes the Sorted node

        Args:
            key (Callable[[Image], Any]): The key to sort images by
            reverse (bool): Whether to reverse the sort order (default is False)
        '''
        super().__init__()
        self.key = key
        self.reverse = reverse

    def eval(self, dataset):
        dataset.images.sort(key=self.key, reverse=self.reverse)
        return dataset


class Shuffle(Node):
    '''Shuffles images in a dataset'''
    def __init__(self):
        '''Initializes the Shuffle node'''
        super().__init__()

    def eval(self, dataset):
        random.shuffle(dataset.images)
        return dataset

class Map(Node):
    '''Maps a function over all images in a dataset'''
    def __init__(self, func: Callable[Image, Image]):
        '''Initializes the Map node

        Args:
            func (Callable[[Image], Image]): The function to map over images
        '''
        super().__init__()
        self.func = func

    def eval(self, dataset):
        dataset.images = [self.func(image) for image in dataset.images]
        return dataset

class Apply(Node):
    '''Applies a function to all images in a dataset'''
    def __init__(self, func: Callable[Image, None]):
        '''Initializes the Apply node

        Args:
            func (Callable[[Image], None]): The function to apply to images
        '''
        super().__init__()
        self.func = func

    def eval(self, dataset):
        for i in dataset.images:
            self.func(i)
        return dataset

class Set(Node):
    '''Sets a property on all images in a dataset'''
    def __init__(self, **kwargs):
        '''Initializes the Set node

        Args:
            **kwargs: The properties to set on the images
        '''
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
