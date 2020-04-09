import hashlib
import logging
import math
import os
import pathlib
from struct import pack, unpack
from typing import Callable, Tuple, Any, List, Dict

import albumentations as albu
import lmdb
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL.Image import Image as Img
from numpy.random.mtrand import RandomState
from pandas.util import hash_pandas_object
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as VTF

from wheel5.random import generate_random_seed
from wheel5.tricks.heatmap import heatmap_to_selection_mask, upsample_heatmap
from .functional import cutmix, mixup, masked_cutmix


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
            arrays = {k: data[k] for k in data.files}
            return NdArrayStorage(arrays)


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


class ImageHeatmapDataset(Dataset):
    def __init__(self, dataset: Dataset, heatmaps: NdArrayStorage, inter_mode: str = 'bilinear'):
        super(ImageHeatmapDataset, self).__init__()
        self.dataset = dataset
        self.heatmaps = heatmaps
        self.inter_mode = inter_mode

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, target, root_index, *rest = self.dataset[index]
        heatmap = torch.from_numpy(self.heatmaps.arrays[str(root_index)])

        w, h = img.size

        heatmap = heatmap.unsqueeze(dim=0)
        heatmap = upsample_heatmap(heatmap, h, w, self.inter_mode)
        heatmap = heatmap.squeeze(dim=0)

        return (img, target, heatmap, root_index, *rest)


class ImageSelectionMaskDataset(Dataset):
    def __init__(self, dataset: Dataset, cutoff_ratio: float):
        super(ImageSelectionMaskDataset, self).__init__()
        self.dataset = dataset
        self.cutoff_ratio = cutoff_ratio

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        (img, target, heatmap, *rest) = self.dataset[index]

        heatmap = heatmap.unsqueeze(dim=0)
        mask = heatmap_to_selection_mask(heatmap, self.cutoff_ratio)
        mask = mask.squeeze(dim=0)

        return (img, target, mask, *rest)


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


class RandomStateDataset(Dataset):
    def __init__(self, name: str = ''):
        super(RandomStateDataset, self).__init__()

        self.name = name
        self.logger = logging.getLogger(f'{__name__}')
        self.debug = self.logger.isEnabledFor(logging.DEBUG)

        self._random_state = None

    @property
    def random_state(self) -> RandomState:
        if not self._random_state:
            seed = generate_random_seed()
            self._random_state = RandomState(seed)

            self.logger.info(f'dataset[{self.name}] - initialized random state [seed={seed}]')

        return self._random_state


class ImageMaskedCutMixDataset(RandomStateDataset):
    def __init__(self, dataset: Dataset, name: str = ''):
        super(ImageMaskedCutMixDataset, self).__init__(name)
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        choice = self.random_state.randint(len(self.dataset))

        img_dst, lb_dst, _, _ = self.dataset[index]
        img_src, lb_src, mask_src, _ = self.dataset[choice]

        img_dst, img_src = VTF.to_tensor(img_dst), VTF.to_tensor(img_src)

        if self.debug:
            self.logger.debug(f'index={index}, choice={choice}')

        img, lb = masked_cutmix(img_src=img_src, lb_src=lb_src,
                                img_dst=img_dst, lb_dst=lb_dst,
                                mask_src=mask_src)

        img = VTF.to_pil_image(img)
        return img, lb, -1


class ImageCutMixDataset(RandomStateDataset):
    def __init__(self, dataset: Dataset, alpha: float, mode: str = 'compact', name: str = ''):
        super(ImageCutMixDataset, self).__init__(name)
        self.dataset = dataset
        self.alpha = alpha
        self.mode = mode

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        choice = self.random_state.randint(len(self.dataset))

        img_dst, lb_dst, _ = self.dataset[index]
        img_src, lb_src, _ = self.dataset[choice]

        img_dst, img_src = VTF.to_tensor(img_dst), VTF.to_tensor(img_src)

        if self.debug:
            self.logger.debug(f'index={index}, choice={choice}')

        img, lb = cutmix(img_src=img_src, lb_src=lb_src,
                         img_dst=img_dst, lb_dst=lb_dst,
                         alpha=self.alpha, mode=self.mode, random_state=self.random_state)

        img = VTF.to_pil_image(img)
        return img, lb, -1


class ImageMixupDataset(RandomStateDataset):
    def __init__(self, dataset: Dataset, alpha: float, name: str = ''):
        super(ImageMixupDataset, self).__init__(name)
        self.dataset = dataset
        self.alpha = alpha

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        choice = self.random_state.randint(len(self.dataset))

        img1, lb1, _ = self.dataset[index]
        img2, lb2, _ = self.dataset[choice]

        img1, img2 = VTF.to_tensor(img1), VTF.to_tensor(img2)

        if self.debug:
            self.logger.debug(f'index={index}, choice={choice}')

        img, lb = mixup(img1, lb1, img2, lb2, self.alpha, random_state=self.random_state)

        img = VTF.to_pil_image(img)
        return img, lb, -1


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable[[Img], Img]):
        super(TransformDataset, self).__init__()

        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, target, *rest = self.dataset[index]
        img = self.transform(img)
        return (img, target, *rest)


class AlbumentationsDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: albu.BasicTransform, use_mask: bool = False):
        super(AlbumentationsDataset, self).__init__()

        self.dataset = dataset
        self.transform = transform

        self.use_mask = use_mask

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.use_mask:
            img, target, mask, *rest = self.dataset[index]
            augmented = self.transform(image=np.array(img), mask=mask.numpy())
            img = Image.fromarray(augmented['image'])
            mask = torch.from_numpy(augmented['mask'])
            return (img, target, mask, *rest)
        else:
            img, target, *rest = self.dataset[index]
            augmented = self.transform(image=np.array(img))
            img = Image.fromarray(augmented['image'])
            return (img, target, *rest)


def targets(dataset: Dataset) -> List[Any]:
    result = []

    for i in range(0, len(dataset)):
        _, target, *_ = dataset[i]
        result.append(target)

    return result
