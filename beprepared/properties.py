from beprepared.workspace import Workspace
from typing import Callable, Any, TypeVar, Generic, Dict
import json
from abc import ABC, abstractmethod

T = TypeVar('T')

class PropertyBag:
    '''Base class for objects that hold properties, like `Image`'''
    def __init__(self, **props) -> None:
        self.props = props

    def with_props(self, props: Dict[str, 'Property']) -> None:
        newprops = self.props.copy()
        for k, v in props.items():
            newprops[k] = v
        return self.__class__(**newprops)

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

    def copy(self):
        return self.__class__(**self.props)


class Property(ABC, Generic[T]):
    def __init__(self) -> None:
        self.owner = None

    @property
    @abstractmethod
    def has_value(self) -> bool:
        pass

    @property
    @abstractmethod
    def value(self) -> T:
        pass

    @value.setter
    @abstractmethod
    def value(self, value: T) -> None:
        pass

def cache_key_tostring(x):
    if isinstance(x, PropertyBag) and x.objectid.has_value:
        x = x.objectid.value
    return json.dumps(x, sort_keys=True, separators=(',','='))

class CachedProperty(Property[T]):
    def __init__(self, scope, *segs):
        super().__init__()
        self._dirty_has_value = True
        self._dirty_value     = True
        if len(segs) == 0:
            self.key = scope
        else:
            self.key = f"{scope}(" + ','.join(cache_key_tostring(x) for x in segs) + ")"

    @property
    def has_value(self) -> bool:
        if self._dirty_has_value:
            ws = Workspace.current
            self._has_value = ws.db.has_prop(self.key)
            self._dirty_has_value = False
        return self._has_value

    @property
    def value(self) -> T:
        if self._dirty_value:
            ws = Workspace.current
            self._value = ws.db.get_prop(self.key)
            self._has_value = self._value is not None
            self._dirty_value = False
            self._dirty_has_value = False
        return self._value

    @value.setter
    def value(self, value: T) -> None:
        self._value = value
        self._has_value = True
        ws = Workspace.current
        ws.db.put_prop(self.key, value)

    def __repr__(self):
        return f"<CachedProperty {self.key} {self._value}>"

NO_VALUE = object()
class ConstProperty(Property[T]):
    def __init__(self, value: T = NO_VALUE):
        super().__init__()
        self._value = value

    @property
    def has_value(self) -> bool:
        return self._value is not NO_VALUE

    @property
    def value(self) -> T:
        return self._value

    @value.setter   
    def value(self, value: T) -> None:
        raise Exception('Const properties cannot be set')

class ComputedProperty(Property[T]):
    def __init__(self, compute: Callable[[Any], T]) -> None:
        self.compute = compute

    @property
    def has_value(self) -> bool:
        return True

    @property
    def value(self) -> T:
        return self.compute(self.owner)

    @value.setter
    def value(self, value: T) -> None:
        raise Exception('Computed properties cannot be set')

__all__ = [ 'PropertyBag', 'Property', 'CachedProperty', 'ConstProperty', 'ComputedProperty' ]
