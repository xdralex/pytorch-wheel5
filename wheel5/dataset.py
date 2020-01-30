import os
import pathlib
from struct import pack, unpack
from typing import Callable, Tuple, List, Optional, NamedTuple

import albumentations as albu
import lmdb
import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import Image as Img
from numpy.random.mtrand import RandomState
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


# TODO: make abstract enough to handle different datasets
# TODO: use transform repr (name or hash) in the path to avoid clashing mistakes?
class LMDBImageDataset(Dataset):
    @staticmethod
    def cached(df: pd.DataFrame, image_dir: str, lmdb_path: str, lmdb_map_size: int = int(8 * (1024 ** 3)), transform: Callable[[Img], Img] = None):
        if os.path.exists(lmdb_path):
            assert os.path.isdir(lmdb_path)
        else:
            LMDBImageDataset.prepare(df, image_dir, lmdb_path, lmdb_map_size, transform)

        return LMDBImageDataset(df, lmdb_path, lmdb_map_size)

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

    def __init__(self, df: pd.DataFrame, lmdb_path: str, lmdb_map_size: int = int(8 * (1024 ** 3))):
        super(LMDBImageDataset, self).__init__()

        self.df = df

        self.df_count = df.shape[0]

        # TODO: bad naming
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

        return image, self.idx_by_name[name], name, index

    def classes(self) -> int:
        return len(self.name_by_idx)


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


class MappingDataset(Dataset):
    def __init__(self, wrapped: Dataset, fn):
        self.wrapped = wrapped
        self.fn = fn

    def __len__(self) -> int:
        return len(self.wrapped)

    def __getitem__(self, index: int):
        return self.fn(self.wrapped[index])


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


def split_eval_main_data(bundle: DataBundle,
                         split: float,
                         eval_batch: int = 256,
                         main_batch: int = 64,
                         eval_workers: int = 4,
                         main_workers: int = 4,
                         random_state: Optional[RandomState] = None) -> (DataBundle, DataBundle):

    if random_state is None:
        random_state = np.random.RandomState()

    eval_indices, main_indices = split_indices(bundle.indices, split=split, random_state=random_state)

    eval_sampler = SubsetRandomSampler(eval_indices)
    main_sampler = SubsetRandomSampler(main_indices)

    eval_loader = DataLoader(bundle.dataset, batch_size=eval_batch, sampler=eval_sampler, num_workers=eval_workers, pin_memory=True)
    main_loader = DataLoader(bundle.dataset, batch_size=main_batch, sampler=main_sampler, num_workers=main_workers, pin_memory=True)

    eval_bundle = DataBundle(loader=eval_loader, dataset=bundle.dataset, indices=eval_indices)
    main_bundle = DataBundle(loader=main_loader, dataset=bundle.dataset, indices=main_indices)

    return eval_bundle, main_bundle
