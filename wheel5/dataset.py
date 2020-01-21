import os
import pathlib
from typing import Callable, Tuple, List, Optional

import lmdb
from struct import pack, unpack
from PIL import Image
from PIL.Image import Image as Img

import pandas as pd
import numpy as np
from numpy.random.mtrand import RandomState

from torch.utils.data import Dataset


class LMDBImageDataset(Dataset):
    @staticmethod
    def cached(df: pd.DataFrame, image_dir: str, lmdb_path: str, lmdb_map_size: int = int(8 * (1024 ** 3)),
               prepare_transform: Callable[[Img], Img] = None,
               getitem_transform: Callable[[Img], Img] = None):

        if os.path.exists(lmdb_path):
            assert os.path.isdir(lmdb_path)
            return LMDBImageDataset(df, lmdb_path, lmdb_map_size, getitem_transform)
        else:
            LMDBImageDataset.prepare(df, image_dir, lmdb_path, lmdb_map_size, prepare_transform)

    @staticmethod
    def prepare(df: pd.DataFrame, image_dir: str, lmdb_path: str, lmdb_map_size: int = int(8 * (1024 ** 3)), transform: Callable[[Img], Img] = None):
        pathlib.Path(lmdb_path).mkdir(parents=True, exist_ok=False)
        with lmdb.open(lmdb_path, map_size=lmdb_map_size, subdir=True) as lmdb_env:
            with lmdb_env.begin(write=True) as txn:
                for row in df.itertuples():
                    image_path = os.path.join(image_dir, row.path)
                    image = Image.open(image_path)

                    if transform is not None:
                        image = transform(image)

                    w, h = image.size

                    k_data = f'data_{row.path}'.encode('ascii')
                    v_data = image.tobytes()
                    txn.put(k_data, v_data)

                    k_meta = f'meta_{row.path}'.encode('ascii')
                    v_meta = pack('HH', w, h)
                    txn.put(k_meta, v_meta)

    def __init__(self, df: pd.DataFrame, lmdb_path: str, lmdb_map_size: int = int(8 * (1024 ** 3)), transform: Callable[[Img], Img] = None):
        self.transform = transform
        self.df = df

        self.df_count = df.shape[0]

        self.name_by_idx = list(np.sort(df['name'].unique()))
        self.idx_by_name = {x: i for i, x in enumerate(self.name_by_idx)}

        self.lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, meminit=False, readahead=False, map_size=lmdb_map_size, subdir=True)

    def __len__(self) -> int:
        return self.df_count

    def __getitem__(self, index: int) -> Tuple[Img, int, str, int]:
        row = self.df.iloc[index, :]

        name = row['name']

        with self.lmdb_env.begin(write=False) as txn:
            k_data = f'data_{row.path}'.encode('ascii')
            v_data = txn.get(k_data)

            k_meta = f'meta_{row.path}'.encode('ascii')
            v_meta = txn.get(k_meta)

            w, h = unpack('HH', v_meta)
            image = Image.frombytes('RGB', (w, h), v_data)

        if self.transform:
            image = self.transform(image)

        return image, self.idx_by_name[name], name, index

    def classes(self) -> int:
        return len(self.name_by_idx)


class WrappingTransformDataset(Dataset):
    def __init__(self, wrapped: Dataset, transform_fn):
        self.wrapped = wrapped
        self.transform_fn = transform_fn

    def __len__(self) -> int:
        return len(self.wrapped)

    def __getitem__(self, index: int):
        return self.transform_fn(self.wrapped[index])


def split_indices(indices: List[int], split: float, random_state: Optional[RandomState] = None) -> (List[int], List[int]):
    if random_state is None:
        random_state = np.random.RandomState()

    shuffled_indices = indices.copy()
    random_state.shuffle(shuffled_indices)

    divider = int(np.round(split * len(indices)))
    return shuffled_indices[:divider], shuffled_indices[divider:]
