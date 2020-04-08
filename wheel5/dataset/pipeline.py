import hashlib
import logging
import os
import pathlib
from struct import pack, unpack

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
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as VTF
from typing import Callable, Tuple, Any, List, Dict

from wheel5.random import generate_random_seed
from .functional import cutmix, mixup


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

    def __getitem__(self, index: int) -> Tuple[Img, Any, int]:
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


class NdArrayStorage(object):
    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.arrays = arrays

    def save(self, path: str):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        path = os.path.join(path, 'data.npz')
        np.savez(path, **self.arrays)

    @staticmethod
    def load(path: str) -> 'NdArrayStorage':
        path = os.path.join(path, 'data.npz')
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.file}
            return NdArrayStorage(arrays)


class ImageTargetDataset(Dataset):
    def __init__(self, dataset: Dataset):
        super(ImageTargetDataset, self).__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Img, Any]:
        img, target, *_ = self.dataset[index]
        return img, target


class ImageOneHotDataset(Dataset):
    def __init__(self, dataset: Dataset, num_classes: int):
        super(ImageOneHotDataset, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, target, *rest = self.dataset[index]

        target = torch.tensor(target)
        lb = F.one_hot(target, self.num_classes).type(torch.float)

        return (img, lb, *rest)


class ImageCutMixDataset(Dataset):
    def __init__(self, dataset: Dataset, alpha: float, mode: str = 'compact', name: str = ''):
        super(ImageCutMixDataset, self).__init__()
        self.dataset = dataset
        self.alpha = alpha
        self.mode = mode

        self.name = name
        self.logger = logging.getLogger(f'{__name__}')
        self.debug = self.logger.isEnabledFor(logging.DEBUG)

        self._random_state = None

    @property
    def random_state(self) -> RandomState:
        if not self._random_state:
            seed = generate_random_seed()
            self._random_state = RandomState(seed)

            self.logger.info(f'cutmix[{self.name}] - initialized random state [seed={seed}]')

        return self._random_state

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Img, Tensor]:
        choice = self.random_state.randint(len(self.dataset))

        img1, lb1, *_ = self.dataset[index]
        img2, lb2, *_ = self.dataset[choice]

        if self.debug:
            self.logger.debug(f'index={index}, choice={choice}')

        img1, img2 = VTF.to_tensor(img1), VTF.to_tensor(img2)
        img, lb = cutmix(img1, lb1, img2, lb2, self.alpha, self.mode, random_state=self.random_state)
        img = VTF.to_pil_image(img)

        return img, lb


class ImageMixupDataset(Dataset):
    def __init__(self, dataset: Dataset, alpha: float, name: str = ''):
        super(ImageMixupDataset, self).__init__()
        self.dataset = dataset
        self.alpha = alpha

        self.name = name
        self.logger = logging.getLogger(f'{__name__}')
        self.debug = self.logger.isEnabledFor(logging.DEBUG)

        self._random_state = None

    @property
    def random_state(self) -> RandomState:
        if not self._random_state:
            seed = generate_random_seed()
            self._random_state = RandomState(seed)

            self.logger.info(f'mixup[{self.name}] - initialized random state [seed={seed}]')

        return self._random_state

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Img, Tensor]:
        choice = self.random_state.randint(len(self.dataset))

        img1, lb1, *_ = self.dataset[index]
        img2, lb2, *_ = self.dataset[choice]

        if self.debug:
            self.logger.debug(f'index={index}, choice={choice}')

        img1, img2 = VTF.to_tensor(img1), VTF.to_tensor(img2)
        img, lb = mixup(img1, lb1, img2, lb2, self.alpha, random_state=self.random_state)
        img = VTF.to_pil_image(img)

        return img, lb


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable[[Img], Img]):
        super(TransformDataset, self).__init__()

        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Img, ...]:
        img, *rest = self.dataset[index]
        img = self.transform(img)
        return (img, *rest)


class AlbumentationsDataset(TransformDataset):
    def __init__(self, dataset: Dataset, transform: albu.BasicTransform):
        def callable_transform(img: Img) -> Img:
            img_arr = np.array(img)
            aug_arr = transform(image=img_arr)
            return Image.fromarray(aug_arr['image'])

        super(AlbumentationsDataset, self).__init__(dataset, callable_transform)


def targets(dataset: Dataset) -> List[Any]:
    result = []

    for i in range(0, len(dataset)):
        _, target, *_ = dataset[i]
        result.append(target)

    return result
