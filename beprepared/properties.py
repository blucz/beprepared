from beprepared.workspace import Workspace
from typing import Callable, Any, TypeVar, Generic
import json

T = TypeVar('T')

class Property(Generic[T]):
    def __init__(self) -> None:
        self._value = None
        self._has_value = False
        self.owner = None

    @property
    def has_value(self) -> bool:
        return self._has_value

    @property
    def value(self) -> T:
        if not self._has_value: 
            return None
        return self._value

    @value.setter
    def value(self, value: T) -> None:
        self._value = value
        self._has_value = True
        ws = Workspace.current
        ws.db.put_prop(self.key, value)


def cache_key_tostring(x):
    if x.__class__.__name__ == 'Image': # TODO: find a cleaner way to do this without a circular import
        x = x.objectid.value
    return json.dumps(x, sort_keys=True, separators=(',','='))

class CachedProperty(Property[T]):
    def __init__(self, scope, *segs) -> None:
        super().__init__()
        if len(segs) == 0:
            self.key = scope
        else:
            self.key = f"{scope}(" + ','.join(cache_key_tostring(x) for x in segs) + ")"
        ws = Workspace.current
        if self.key is not None:
            cached_value = ws.db.get_prop(self.key)
            if cached_value is not None:
                self._value = cached_value
                self._has_value = True

    def __repr__(self):
        return f"<CachedProperty {self.key} {self._value}>"

NO_VALUE = object()
class ConstProperty(Property[T]):
    def __init__(self, value: T = NO_VALUE):
        super().__init__()
        if value is NO_VALUE:
            self._has_value = False
            self._value = None
        else:
            self._has_value = True
            self._value = value

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
