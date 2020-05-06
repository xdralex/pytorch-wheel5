import os
import pathlib
import pickle
from typing import TypeVar, List

import lmdb
import numpy as np

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lmdb_env.close()


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
