from beprepared.properties import Property, ConstProperty
from typing import Dict

class Image:
    ALLOWED_FORMATS = {'JPEG', 'PNG', 'WEBP', 'GIF', 'TIFF', 'BMP' }

    def __init__(self, **props) -> None:
        self.props = props

    def with_props(self, props: Dict[str, Property]) -> None:
        newprops = self.props.copy()
        for k, v in props.items():
            newprops[k] = v
        return Image(**newprops)

    def __setattr__(self, name, value):
        if name == "props":
            self.__dict__[name] = value
            return
        if isinstance(value, Property):
            self.props[name] = value
        else:
            self.props[name] = ConstProperty(value)
        self.props[name].owner = self

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return self.props.get(name) or ConstProperty()

    def copy(self) -> 'Image':
        return Image(**self.props)
