import json
import os
import pathlib
from typing import Dict, TypeVar, Generic, List

import lmdb
import numpy as np


T = TypeVar('T')


class NdArraysStorage(object):
    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.arrays = arrays

    def save(self, path: str):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        path = os.path.join(path, 'data.npz')
        np.savez(path, **self.arrays)

    @staticmethod
    def load(path: str) -> 'NdArraysStorage':
        path = os.path.join(path, 'data.npz')
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
            return NdArraysStorage(arrays)


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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lmdb_env.close()


class DictStorage(Generic[T]):
    def __init__(self, data: Dict[str, T]):
        self.data = data

    def save(self, path: str, compact: bool = True):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        path = os.path.join(path, 'data.json')
        with open(path, mode='x') as f:
            json.dump(self.data, f, indent=None if compact else 4)

    @staticmethod
    def load(path: str) -> 'DictStorage[T]':
        path = os.path.join(path, 'data.json')
        with open(path, mode='r') as f:
            return json.load(f)


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
