# Hacking

## Coding Standards

Write pythonic python, declare types at all API surfaces and use google-style docstrings.

## Creating new Nodes

This is an example node that is defined correctly:

```python
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

    def eval(self, dataset: Dataset) -> Dataset:
        for image in dataset.images:
            if not image.tags.has_value:
                image.tags = ConstProperty(set(self.tags))
                continue
            else:
                image.tags = ConstProperty(image.tags.value.union(self.tags))
        return dataset
```

Follow this general pattern and things should be great. Note that `eval` can also accept `datasets: List[Dataset]`. This is rare, but required for certain nodes like `Concat`.

## How nodes are executed

Beprepared nodes are executed sequentially in dependency order. While this may be inefficient in some cases, it's necessary because nodes 
often compete for limited resources, whether that's a GPU or a Human who needs to perform an operation. This may change in the future, 
but for now, nodes run sequentially.

Before `eval` is invoked, the dataset and images are deep-copied. This gives the code functional semantics while 
enabling the code itself to be written in the imperative style that's more typical in python. It is safe
to "flow" the same dataset into multiple nodes, because each node gets its own copy of the dataset and the images within, 
and images and datasets may be freely mutated within `eval`

## The Web Interface

At startup, `beprepared` launches a web interface on `0.0.0.0:8989`.

When a node needs a web interface, it supplies an `Applet` to `workspace.web`. This applet "takes over" the main area of the web interface.

Applets are defined in Python using FastAPI and implemented in [Vue3](https://vuejs.org) + [Bootstrap](https://getbootstrap.com) on the 
client side. If you are hacking on the web stuff, you will either need to `npm run build` in `web/` after making changes, or `npm run dev` 
and then set up a `.env` with `VITE_API_URL=http://IP_ADDRESS:8989`.

By convention, each applet is a vue component in `beprepared/web/`. On the python side, you can define `FastAPI` endpoints to serve the applet.

See `beprepared/nodes/humantag.py` for an example of how to define an applet and use it in a node.


