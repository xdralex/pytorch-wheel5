import os
import pathlib
from struct import pack, unpack
from typing import Callable, Tuple, List, Optional, NamedTuple, Any

import albumentations as albu
import lmdb
import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import Image as Img
from numpy.random.mtrand import RandomState
from torch.utils.data import Dataset, DataLoader


class LMDBImageDataset(Dataset):
    r"""A dataset storing images in the LMDB store.

    This dataset takes a user-provided dataframe ['path', 'target', ...] to
    locate the images. The 'path' column in the dataframe must point to
    the image filename, and the 'target' column must contain the target value.
    """

    @staticmethod
    def cached(df: pd.DataFrame,
               image_dir: str,
               lmdb_path: str,
               lmdb_map_size: int = int(8 * (1024 ** 3)),
               transform: Callable[[Img], Img] = None,
               check_transform: bool = True):

        if os.path.exists(lmdb_path):
            if not os.path.isdir(lmdb_path):
                raise Exception(f'LMDB path {lmdb_path} must be a directory')

            if check_transform:
                with open(os.path.join(lmdb_path, '.transform'), mode='r') as f:
                    lmdb_transform_str = f.read().strip()
                    if lmdb_transform_str != str(transform):
                        raise Exception(f'Transform used in LMDB is different from the supplied transform:\n' +
                                        f'lmdb transform: {lmdb_transform_str}\n' +
                                        f'supplied transform: {transform}')
        else:
            LMDBImageDataset.prepare(df, image_dir, lmdb_path, lmdb_map_size, transform)

        return LMDBImageDataset(df, lmdb_path, lmdb_map_size)

    @staticmethod
    def prepare(df: pd.DataFrame,
                image_dir: str,
                lmdb_path: str,
                lmdb_map_size: int = int(8 * (1024 ** 3)),
                transform: Callable[[Img], Img] = None,
                write_transform: bool = True):

        pathlib.Path(lmdb_path).mkdir(parents=True, exist_ok=False)

        if write_transform:
            with open(os.path.join(lmdb_path, '.transform'), mode='x') as f:
                f.write(f'{transform}\n')

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

    def __init__(self,
                 df: pd.DataFrame,
                 lmdb_path: str,
                 lmdb_map_size: int = int(8 * (1024 ** 3))):
        super(LMDBImageDataset, self).__init__()

        self.df = df
        self.lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, meminit=False, readahead=False, map_size=lmdb_map_size, subdir=True)

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> Tuple[Img, Any, int]:
        row = self.df.iloc[index, :]
        target = row['target']

        with self.lmdb_env.begin(write=False) as txn:
            k_data = f'data_{row.path}'.encode('ascii')
            v_data = txn.get(k_data)

            k_meta = f'meta_{row.path}'.encode('ascii')
            v_meta = txn.get(k_meta)

            w, h = unpack('HH', v_meta)
            image = Image.frombytes('RGB', (w, h), v_data)

        return image, target, index


class TransformDataset(Dataset):
    def __init__(self, wrapped: Dataset, transform: Callable[[Img], Img]):
        super(TransformDataset, self).__init__()

        self.wrapped = wrapped
        self.transform = transform

    def __len__(self) -> int:
        return len(self.wrapped)

    def __getitem__(self, index: int):
        item_tuple = self.wrapped[index]
        item_list = list(item_tuple)

        image = item_list[0]
        image = self.transform(image)
        item_list[0] = image

        return tuple(item_list)


class AlbumentationsDataset(TransformDataset):
    def __init__(self, wrapped: Dataset, transform: albu.BasicTransform):
        def callable_transform(image: Img) -> Img:
            image_arr = np.array(image)
            aug_arr = transform(image=image_arr)
            return Image.fromarray(aug_arr['image'])

        super(AlbumentationsDataset, self).__init__(wrapped, callable_transform)


class DataBundle(NamedTuple):
    loader: DataLoader
    dataset: Dataset
    indices: List[int]


def split_indices(indices: List[int], split: float, random_state: Optional[RandomState] = None) -> (List[int], List[int]):
    if random_state is None:
        random_state = np.random.RandomState()

    shuffled_indices = indices.copy()
    random_state.shuffle(shuffled_indices)

    divider = int(np.round(split * len(indices)))
    return shuffled_indices[:divider], shuffled_indices[divider:]
