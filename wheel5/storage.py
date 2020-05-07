import os
import pathlib
import pickle
from typing import TypeVar, List, Tuple

import lmdb
import numpy as np

from wheel5.tasks.detection import BoundingBox

T = TypeVar('T')


class LMDBDict(object):
    def __init__(self, lmdb_path: str, lmdb_map_size: int = int(8 * (1024 ** 3))):
        if os.path.exists(lmdb_path):
            if not os.path.isdir(lmdb_path):
                raise Exception(f'LMDB path {lmdb_path} must be a directory')
        else:
            pathlib.Path(lmdb_path).mkdir(parents=True, exist_ok=False)

        self.lmdb_env = lmdb.open(lmdb_path, map_size=lmdb_map_size, subdir=True)

    def __contains__(self, key: str) -> bool:
        with self.lmdb_env.begin(write=False) as txn:
            result = txn.get(key.encode('ascii'))
            return result is not None

    def __getitem__(self, key: str) -> bytes:
        with self.lmdb_env.begin(write=False) as txn:
            result = txn.get(key.encode('ascii'))
            if result is None:
                raise KeyError('Key {k} not found')
            return result

    def __setitem__(self, key: str, value: bytes):
        with self.lmdb_env.begin(write=True) as txn:
            txn.put(key.encode('ascii'), value, dupdata=False, overwrite=True)

    def __delitem__(self, key: str):
        with self.lmdb_env.begin(write=True) as txn:
            txn.delete(key.encode('ascii'))

    def __iter__(self):
        with self.lmdb_env.begin(write=False) as txn:
            with txn.cursor() as cursor:
                for key, _ in cursor:
                    yield key.decode('ascii')
        return

    def keys(self) -> List[str]:
        with self.lmdb_env.begin(write=False) as txn:
            with txn.cursor() as cursor:
                return list([key.decode('ascii') for key, _ in cursor])

    def values(self) -> List[bytes]:
        with self.lmdb_env.begin(write=False) as txn:
            with txn.cursor() as cursor:
                return list([value for _, value in cursor])

    def items(self) -> List[Tuple[str, bytes]]:
        with self.lmdb_env.begin(write=False) as txn:
            with txn.cursor() as cursor:
                return list([(key.decode('ascii'), value) for key, value in cursor])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lmdb_env.close()


class HeatmapLMDBDict(object):
    def __init__(self, lmdb_dict: LMDBDict):
        self.lmdb_dict = lmdb_dict

    def __contains__(self, key: str) -> bool:
        return key in self.lmdb_dict

    def __getitem__(self, key: str) -> np.ndarray:
        data = self.lmdb_dict[key]
        return decode_ndarray(data)

    def __setitem__(self, key: str, value: np.ndarray):
        data = encode_ndarray(value)
        self.lmdb_dict[key] = data

    def __delitem__(self, key: str):
        del self.lmdb_dict[key]

    def __iter__(self):
        yield from self.lmdb_dict

    def keys(self) -> List[str]:
        return self.lmdb_dict.keys()

    def values(self) -> List[np.ndarray]:
        return [decode_ndarray(value) for value in self.lmdb_dict.values()]

    def items(self) -> List[Tuple[str, np.ndarray]]:
        return [(key, decode_ndarray(value)) for key, value in self.lmdb_dict.items()]


class BoundingBoxesLMDBDict(object):
    def __init__(self, lmdb_dict: LMDBDict):
        self.lmdb_dict = lmdb_dict

    def __contains__(self, key: str) -> bool:
        return key in self.lmdb_dict

    def __getitem__(self, key: str) -> List[BoundingBox]:
        data = self.lmdb_dict[key]
        return self._decode_bboxes(data)

    def __setitem__(self, key: str, value: List[BoundingBox]):
        data = self._encode_bboxes(value)
        self.lmdb_dict[key] = data

    def __delitem__(self, key: str):
        del self.lmdb_dict[key]

    def __iter__(self):
        yield from self.lmdb_dict

    def keys(self) -> List[str]:
        return self.lmdb_dict.keys()

    def values(self) -> List[List[BoundingBox]]:
        return [self._decode_bboxes(value) for value in self.lmdb_dict.values()]

    def items(self) -> List[Tuple[str, List[BoundingBox]]]:
        return [(key, self._decode_bboxes(value)) for key, value in self.lmdb_dict.items()]

    @staticmethod
    def _encode_bboxes(value: List[BoundingBox]) -> bytes:
        return encode_list([bbox.encode() for bbox in value], size=BoundingBox.byte_size())

    @staticmethod
    def _decode_bboxes(data: bytes) -> List[BoundingBox]:
        return [BoundingBox.decode(b) for b in decode_list(data, size=BoundingBox.byte_size())]


def encode_list(lst: List[bytes], size: int) -> bytes:
    for x in lst:
        assert len(x) == size

    return b''.join(lst)


def decode_list(b: bytes, size: int) -> List[bytes]:
    assert len(b) % size == 0

    lst = []
    num_chunks = len(b) // size
    for i in range(0, num_chunks):
        chunk = b[i * size: (i + 1) * size]
        lst.append(chunk)

    return lst


def encode_ndarray(arr: np.ndarray) -> bytes:
    return pickle.dumps(arr)


def decode_ndarray(b: bytes) -> np.ndarray:
    return pickle.loads(b)
