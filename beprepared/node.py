from beprepared.dataset import Dataset
from beprepared.workspace import Workspace

import inspect
import logging
import time

class NodeMeta(type):
    '''This metaclass enables workflow DSL patterns like `Concat << a << b << c` without requiring `Concat() << a << b << c`.'''
    def __rshift__(cls, other):
        obj = cls()
        return obj >> other
    def __lshift__(cls, other):
        obj = cls()
        return obj << other

class Node(metaclass=NodeMeta):
    def __init__(self) -> None:
        self.sources = []
        self.sinks = []
        self.workspace = Workspace.current
        self.log = logging.getLogger(
            f"{self.workspace.log.name}.{self.__class__.__name__}"
        )
        self.workspace.nodes.append(self)

    def eval(self) -> Dataset:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    @property
    def source(self):
        if len(self.sources) > 1:
            raise Exception("Node has multiple sources")
        if len(self.sources) == 0:
            raise Exception("Node has no sources")
        return self.sources[0]

    def __rshift__(self, other):
        if isinstance(other, type):
            other = other()
        self.sinks.append(other)
        other.sources.append(self)
        return other

    def __lshift__(self, other):
        if isinstance(other, type):
            other = other()
        self.sources.append(other)
        other.sinks.append(self)
        return self

    def __call__(self, *args, **kwargs):
        datasets = [s().copy() for s in self.sources]
        params = inspect.signature(self.eval).parameters
        evalfn = None
        if "dataset" in params and "datasets" not in params:
            if len(datasets) == 1:
                evalfn = lambda: self.eval(dataset=datasets[0])
            else:
                raise ValueError(
                    "The method expects a single source, but multiple were provided."
                )
        elif "datasets" in params:
            evalfn = lambda: self.eval(datasets=datasets)
        elif len(datasets) == 0:
            evalfn = lambda: self.eval()
        else:
            raise ValueError(
                f"The eval method for {self.__class__.__name__} must take either `source` or `sources` as an argument."
            )

        self.workspace.web.connect_log(self.__class__.__name__, self.log)
        self.log.info("Start")
        start_time = time.perf_counter()
        ret = evalfn()
        end_time = time.perf_counter()
        self.log.info(f"Finished in {end_time - start_time:.2f}s")
        self.workspace.web.disconnect_log(self.__class__.__name__, self.log)
        
        if ret is None:
            raise ValueError(f"Node {self.__class__.__name__} returned None. Nodes must return a Dataset object.")
        return ret
