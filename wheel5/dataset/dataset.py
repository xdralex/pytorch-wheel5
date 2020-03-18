from abc import ABC, abstractmethod
import hashlib
import os
import pathlib
from struct import pack, unpack
from typing import Callable, Tuple, Any, List, TypeVar, Generic, Optional

import albumentations as albu
import lmdb
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL.Image import Image as Img
from numpy.random.mtrand import RandomState
from pandas.util import hash_pandas_object
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torch.nn import functional as F
from torchvision.transforms import functional as VTF


from .functional import cutmix, mixup


T = TypeVar('T')


class ImageDataset(ABC, Generic[T], Dataset):
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Img, T, int]:
        pass


class LMDBImageDataset(ImageDataset[T]):
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
               check_fingerprint: bool = True):

        if os.path.exists(lmdb_path):
            if not os.path.isdir(lmdb_path):
                raise Exception(f'LMDB path {lmdb_path} must be a directory')

            if check_fingerprint:
                with open(os.path.join(lmdb_path, '.fingerprint'), mode='r') as f:
                    lmdb_fingerprint = f.read().strip()
                    supplied_fingerprint = LMDBImageDataset.fingerprint(df, image_dir, transform)
                    if lmdb_fingerprint != supplied_fingerprint:
                        raise Exception(f'Fingerprint used in LMDB is different from the supplied fingerprint:\n' +
                                        f'lmdb fingerprint: {lmdb_fingerprint}\n' +
                                        f'supplied fingerprint: {supplied_fingerprint}')
        else:
            LMDBImageDataset.prepare(df, image_dir, lmdb_path, lmdb_map_size, transform)

        return LMDBImageDataset(df, lmdb_path, lmdb_map_size)

    @staticmethod
    def prepare(df: pd.DataFrame,
                image_dir: str,
                lmdb_path: str,
                lmdb_map_size: int = int(8 * (1024 ** 3)),
                transform: Callable[[Img], Img] = None):

        pathlib.Path(lmdb_path).mkdir(parents=True, exist_ok=False)

        with open(os.path.join(lmdb_path, '.fingerprint'), mode='x') as f:
            f.write(f'{LMDBImageDataset.fingerprint(df, image_dir, transform)}\n')

        with lmdb.open(lmdb_path, map_size=lmdb_map_size, subdir=True) as lmdb_env:
            with lmdb_env.begin(write=True) as txn:
                for index, row in enumerate(df.itertuples()):
                    image_path = os.path.join(image_dir, row.path)
                    image = Image.open(image_path)

                    if transform is not None:
                        image = transform(image)

                    w, h = image.size

                    k_data = f'data#{index}'.encode('ascii')
                    v_data = image.tobytes()
                    txn.put(k_data, v_data)

                    k_meta = f'meta#{index}'.encode('ascii')
                    v_meta = pack('HH', w, h)
                    txn.put(k_meta, v_meta)

    @staticmethod
    def fingerprint(df: pd.DataFrame, image_dir: str, transform: Callable[[Img], Img]) -> str:
        df_hash = hashlib.sha256(hash_pandas_object(df, index=True).values).hexdigest()
        return f'{df_hash}\n{image_dir}\n{transform}'

    def __init__(self,
                 df: pd.DataFrame,
                 lmdb_path: str,
                 lmdb_map_size: int = int(8 * (1024 ** 3))):
        super(LMDBImageDataset, self).__init__()

        self.df = df
        self.lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, meminit=False, readahead=False, map_size=lmdb_map_size, subdir=True)

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> Tuple[Img, T, int]:
        row = self.df.iloc[index, :]
        target = row['target']

        with self.lmdb_env.begin(write=False) as txn:
            k_data = f'data#{index}'.encode('ascii')
            v_data = txn.get(k_data)

            k_meta = f'meta#{index}'.encode('ascii')
            v_meta = txn.get(k_meta)

            w, h = unpack('HH', v_meta)
            image = Image.frombytes('RGB', (w, h), v_data)

        return image, target, index

    def targets(self) -> List[T]:
        return self.df['target'].tolist()


class ImageOneHotDataset(ImageDataset[Tensor]):
    def __init__(self, dataset: ImageDataset[int], num_classes: int):
        super(ImageOneHotDataset, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Img, Tensor, int]:
        img, target, index = self.dataset[index]
        lb = F.one_hot(target, self.num_classes).type(torch.float)
        return img, lb, index


class ImageCutMixDataset(ImageDataset[Tensor]):
    def __init__(self, dataset: ImageOneHotDataset, alpha: float, mode: str = 'compact', random_state: Optional[RandomState] = None):
        super(ImageCutMixDataset, self).__init__()
        self.dataset = dataset
        self.alpha = alpha
        self.mode = mode
        self.random_state = random_state or RandomState()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Img, Tensor, int]:
        img1, lb1, index = self.dataset[index]
        img2, lb2, _ = self.dataset[self.random_state.randint(len(self.dataset))]

        img1, img2 = VTF.to_tensor(img1), VTF.to_tensor(img2)
        img, lb = cutmix(img1, lb1, img2, lb2, self.alpha, self.mode, self.random_state)
        img = VTF.to_pil_image(img)

        return img, lb, -1


class ImageMixupDataset(ImageDataset[Tensor]):
    def __init__(self, dataset: ImageOneHotDataset, alpha: float, random_state: Optional[RandomState] = None):
        super(ImageMixupDataset, self).__init__()
        self.dataset = dataset
        self.alpha = alpha
        self.random_state = random_state or RandomState()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Img, Tensor, int]:
        img1, lb1, index = self.dataset[index]
        img2, lb2, _ = self.dataset[self.random_state.randint(len(self.dataset))]

        img1, img2 = VTF.to_tensor(img1), VTF.to_tensor(img2)
        img, lb = mixup(img1, lb1, img2, lb2, self.alpha, self.random_state)
        img = VTF.to_pil_image(img)

        return img, lb, -1


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


class SequentialSubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
