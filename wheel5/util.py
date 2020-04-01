from typing import TypeVar, Generic, Optional

T = TypeVar('T')


class Closure(Generic[T]):
    def __init__(self):
        self.value: Optional[T] = None
