import collections
from typing import TypeVar, Optional, Union, List

T = TypeVar('T')


def as_list(x: Optional[Union[T, List[T]]]) -> List[T]:
    if x is None:
        return []
    elif not isinstance(x, collections.Iterable):
        return [x]
    else:
        return x
