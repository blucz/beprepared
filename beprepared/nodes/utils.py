from beprepared.properties import ConstProperty
from beprepared.node import Node
from beprepared.dataset import Dataset
from beprepared.image import Image
from beprepared.workspace import Abort
import numpy as np
from tqdm import tqdm as _tqdm
import time
import threading
from contextlib import contextmanager
from colorama import Fore, Style
from typing import List, Any, Callable, Optional
import random
import shutil
import textwrap
import os
from beprepared.workspace import Workspace

class WebTqdm(_tqdm):
    """tqdm that can print to terminal and web interface"""
    def __init__(self, *args, **kwargs):
        self.web = Workspace.current.web if Workspace.current else None
        self._last_line = ""
        self._progress_active = False
        super().__init__(*args, **kwargs)
        
    def display(self, msg=None, pos=None):
        super().display(msg, pos)
        if self.web:
            if not self.disable and not self.total or self.n < self.total:
                # Only send progress if not disabled and not complete
                msg = {
                    'command': 'progress',
                    'desc': self.desc or '',
                    'n': self.n,
                    'total': self.total if hasattr(self, 'total') else None,
                    'rate': self.format_dict.get('rate', 0),
                    'elapsed': self.format_dict.get('elapsed', 0),
                    'elapsed': self.format_dict.get('elapsed', 0),
                }
                self.web.broadcast(msg)
                self._progress_active = True
            elif self._progress_active:  # Clear progress when done
                self.web.broadcast({'command': 'progress', 'clear': True})
                self._progress_active = False

def tqdm(*args, **kwargs):
    """Drop-in replacement for tqdm that also sends updates to web interface"""
    return WebTqdm(*args, **kwargs)

class Concat(Node):
    '''Concatenates multiple datasets into one.

    Because Python operator overloading is limited, typical usage is as follows:

    set_a = Load("dir_a")
    set_b = Load("dir_b")
    set_c = Load("dir_c")

    (Concat << set_a << set_b << set_c) >> Save("output_dir")
    '''
    def __init__(self, *nodes):
        '''Initializes the Concat node

        Args:
            *nodes: The datasets to concatenate
        '''
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

    For larger datasets, the `Save` node generates an `index.html` file in the output directory which can be loaded in a web browser to see the dataset more easily.'''
    def __init__(self, include_hidden_properties=False):
        '''Initializes the Info node

        Args:
            include_hidden_properties (bool): Whether to include hidden properties (default is False)
        '''
        super().__init__()
        self.include_hidden_properties = include_hidden_properties

    def eval(self, dataset) -> Dataset:
        if len(dataset.images) == 0:
            return dataset
        max_length_of_propname = max(len(k) for image in dataset.images for k in image.props.keys())
        terminal_cols = shutil.get_terminal_size().columns 
        for image in dataset.images:
            last_was_multiline = False
            print("-" * terminal_cols)
            for k,v in image.props.items():
                if k.startswith('_') and not self.include_hidden_properties:
                    continue
                if not v.has_value: 
                    continue
                value = v.value
                if isinstance(value, np.ndarray):
                    s = np.array2string(value, threshold=50)
                elif hasattr(value, 'show'):
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
    '''Takes the `n` images from a dataset. By default, takes the first N, unless you enable `random`. 
       Setting `random` + `seed` allows you to repeatedly sample the same items, which can be useful for 
       taste-testing processing steps on a small subset of a large dataset before paying the time/cost of 
       processing the whole thing.'''
    def __init__(self, n: int, random: bool=False, seed: int=None):
        '''Initializes the Take node

        Args:
            n (int): The number of images to take
            random (bool): Whether to take the images randomly (default is False)
            seed (int): The seed to use for random sampling (default is None)
        '''
        super().__init__()
        self.n = n
        self.random = random
        self.seed = seed

    def eval(self, dataset):
        if self.random:
            rng = random.Random()
            rng.seed(self.seed)
            dataset.images = rng.sample(dataset.images, self.n)
        else:
            dataset.images = dataset.images[:self.n]
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

class Passthrough(Node):
    '''A node that does nothing and returns the dataset unchanged.
    
    This can be useful as a no-op placeholder in conditional pipeline branches.'''
    def __init__(self):
        '''Initializes the Passthrough node'''
        super().__init__()

    def eval(self, dataset):
        return dataset

class MapCaption(Node):
    '''Maps the current caption to a new caption using a function'''
    def __init__(self, func: Callable[[str], str]):
        '''Initializes the MapCaption node

        Args:
            func (Callable[[str], str]): Function that takes the current caption and returns a new caption
        '''
        super().__init__()
        self.func = func

    def eval(self, dataset):
        for image in dataset.images:
            if image.caption.value:
                new_caption = self.func(image.caption.value)
                image.caption = ConstProperty(new_caption)
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


class Fail(Node):
    '''Fails with an error message. This can be useful if you want to interrupt a pipeline before it finishes, or for incomplete work.'''
    def __init__(self, message: str = "error"):
        '''Initializes the Fail node

        Args:
            message (str): The error message to fail with
        '''
        super().__init__()
        self.message = message

    def eval(self, dataset):
        raise Abort(self.message)

class Sleep(Node):
    '''Sleeps for some amount of time'''
    def __init__(self, seconds: float):
        '''Initializes the Sleep node

        Args:
            seconds (float): The number of seconds to sleep
        '''
        super().__init__()
        self.seconds = seconds

    def eval(self, dataset):
        if self.seconds < 1:
            time.sleep(self.seconds)
        else:
            for _ in tqdm(range(int(self.seconds)), desc="Sleeping"):
                time.sleep(1)
                self.log.info("sleep...")
        return dataset


