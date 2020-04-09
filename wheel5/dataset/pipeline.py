import hashlib
import logging
import math
import os
import pathlib
from struct import pack, unpack
from typing import Callable, Tuple, Any, List, Dict, Union

import PIL
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


class BaseDataset(Dataset):
    def __init__(self, name: str = ''):
        super(BaseDataset, self).__init__()

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


class LMDBImageDataset(BaseDataset):
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
               check_fingerprint: bool = True,
               name: str = ''):

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

        return LMDBImageDataset(df, lmdb_path, lmdb_map_size, name)

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
                 lmdb_map_size: int = int(8 * (1024 ** 3)),
                 name: str = ''):
        super(LMDBImageDataset, self).__init__(name=name)

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

        if self.debug:
            self.logger.debug(f'dataset[{self.name}] - #{index}: '
                              f'image={shape(image)}, target={target}')

        return image, target, index


class ImageHeatmapDataset(BaseDataset):
    def __init__(self, dataset: Dataset, heatmaps: NdArrayStorage, inter_mode: str = 'bilinear', name: str = ''):
        super(ImageHeatmapDataset, self).__init__(name=name)
        self.dataset = dataset
        self.heatmaps = heatmaps
        self.inter_mode = inter_mode

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, target, native, *rest = self.dataset[index]
        w, h = img.size

        heatmap = torch.from_numpy(self.heatmaps.arrays[str(native)])

        upsampled = heatmap.unsqueeze(dim=0)
        upsampled = upsample_heatmap(upsampled, h, w, self.inter_mode)
        upsampled = upsampled.squeeze(dim=0)

        if self.debug:
            self.logger.debug(f'dataset[{self.name}] - #{index}[{native}]: '
                              f'image={shape(img)}, target={target}, heatmap={shape(heatmap)}, upsampled={shape(upsampled)}')

        return (img, target, native, upsampled, *rest)


class ImageSelectionMaskDataset(BaseDataset):
    def __init__(self, dataset: Dataset, cutoff_ratio: float, name: str = ''):
        super(ImageSelectionMaskDataset, self).__init__(name=name)
        self.dataset = dataset
        self.cutoff_ratio = cutoff_ratio

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        (img, target, native, heatmap, *rest) = self.dataset[index]

        heatmap_expanded = heatmap.unsqueeze(dim=0)
        mask = heatmap_to_selection_mask(heatmap_expanded, self.cutoff_ratio)
        mask = mask.squeeze(dim=0)

        if self.debug:
            self.logger.debug(f'dataset[{self.name}] - #{index}[{native}]: '
                              f'image={shape(img)}, target={target}, heatmap={shape(heatmap)}, mask={shape(mask)}')

        return (img, target, native, mask, *rest)


class ImageOneHotDataset(BaseDataset):
    def __init__(self, dataset: Dataset, num_classes: int, name: str = ''):
        super(ImageOneHotDataset, self).__init__(name=name)
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, target, native, *rest = self.dataset[index]
        lb = F.one_hot(torch.tensor(target), self.num_classes).type(torch.float)

        if self.debug:
            self.logger.debug(f'dataset[{self.name}] - #{index}[{native}]: '
                              f'image={shape(img)}, target={target}, lb={lb}')

        return (img, lb, native, *rest)


class ImageMaskedCutMixDataset(BaseDataset):
    def __init__(self, dataset: Dataset, name: str = ''):
        super(ImageMaskedCutMixDataset, self).__init__(name)
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        choice = self.random_state.randint(len(self.dataset))

        img_dst, lb_dst, native_dst, _ = self.dataset[index]
        img_src, lb_src, native_src, mask_src = self.dataset[choice]

        native = -1
        img_dst, img_src = VTF.to_tensor(img_dst), VTF.to_tensor(img_src)
        img, lb, weight = masked_cutmix(img_src=img_src, lb_src=lb_src,
                                        img_dst=img_dst, lb_dst=lb_dst,
                                        mask_src=mask_src)

        if self.debug:
            self.logger.debug(f'dataset[{self.name}]\n'
                              f'    dst #{index}[{native_dst}]: image={shape(img_dst)}, lb={lb_dst}\n'
                              f'    src #{choice}[{native_src}]: image={shape(img_src)}, lb={lb_src}\n'
                              f'    mix #{index}[{native}]: image={shape(img)}, lb={lb}, weight={weight}')

        img = VTF.to_pil_image(img)
        return img, lb, native


class ImageCutMixDataset(BaseDataset):
    def __init__(self, dataset: Dataset, alpha: float, mode: str = 'compact', name: str = ''):
        super(ImageCutMixDataset, self).__init__(name)
        self.dataset = dataset
        self.alpha = alpha
        self.mode = mode

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        choice = self.random_state.randint(len(self.dataset))

        img_dst, lb_dst, native_dst = self.dataset[index]
        img_src, lb_src, native_src = self.dataset[choice]

        native = -1
        img_dst, img_src = VTF.to_tensor(img_dst), VTF.to_tensor(img_src)
        img, lb, weight = cutmix(img_src=img_src, lb_src=lb_src,
                                 img_dst=img_dst, lb_dst=lb_dst,
                                 alpha=self.alpha, mode=self.mode, random_state=self.random_state)

        if self.debug:
            self.logger.debug(f'dataset[{self.name}]\n'
                              f'    dst #{index}[{native_dst}]: image={shape(img_dst)}, lb={lb_dst}\n'
                              f'    src #{choice}[{native_src}]: image={shape(img_src)}, lb={lb_src}\n'
                              f'    mix #{index}[{native}]: image={shape(img)}, lb={lb}, weight={weight}')

        img = VTF.to_pil_image(img)
        return img, lb, native


class ImageMixupDataset(BaseDataset):
    def __init__(self, dataset: Dataset, alpha: float, name: str = ''):
        super(ImageMixupDataset, self).__init__(name)
        self.dataset = dataset
        self.alpha = alpha

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        choice = self.random_state.randint(len(self.dataset))

        img_dst, lb_dst, native_dst = self.dataset[index]
        img_src, lb_src, native_src = self.dataset[choice]

        img_dst, img_src = VTF.to_tensor(img_dst), VTF.to_tensor(img_src)
        img, lb, weight = mixup(img_src=img_src, lb_src=lb_src,
                                img_dst=img_dst, lb_dst=lb_dst,
                                alpha=self.alpha, random_state=self.random_state)

        native = -1
        if self.debug:
            self.logger.debug(f'dataset[{self.name}]\n'
                              f'    dst #{index}[{native_dst}]: image={shape(img_dst)}, lb={lb_dst}\n'
                              f'    src #{choice}[{native_src}]: image={shape(img_src)}, lb={lb_src}\n'
                              f'    mix #{index}[{native}]: image={shape(img)}, lb={lb}, weight={weight}')

        img = VTF.to_pil_image(img)
        return img, lb, native


class TransformDataset(BaseDataset):
    def __init__(self, dataset: Dataset, transform: Callable[[Img], Img], name: str = ''):
        super(TransformDataset, self).__init__(name)
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, target, native, *rest = self.dataset[index]
        img_aug = self.transform(img)

        if self.debug:
            self.logger.debug(f'dataset[{self.name}] - #{index}[{native}]: '
                              f'image={shape(img)}, image^={shape(img_aug)}, target={target}')

        return (img_aug, target, native, *rest)


class AlbumentationsDataset(BaseDataset):
    def __init__(self, dataset: Dataset, transform: albu.BasicTransform, name: str = '', use_mask: bool = False):
        super(AlbumentationsDataset, self).__init__(name)
        self.dataset = dataset
        self.transform = transform
        self.use_mask = use_mask

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.use_mask:
            img, target, native, mask, *rest = self.dataset[index]

            augmented = self.transform(image=np.array(img), mask=mask.numpy())
            img_aug = Image.fromarray(augmented['image'])
            mask_aug = torch.from_numpy(augmented['mask'])

            if self.debug:
                self.logger.debug(f'dataset[{self.name}] - #{index}[{native}]: '
                                  f'image={shape(img)}, image^={shape(img_aug)}, mask={shape(mask)}, mask^={shape(mask_aug)}, target={target}')

            return (img_aug, target, native, mask_aug, *rest)
        else:
            img, target, native, *rest = self.dataset[index]
            augmented = self.transform(image=np.array(img))
            img_aug = Image.fromarray(augmented['image'])

            if self.debug:
                self.logger.debug(f'dataset[{self.name}] - #{index}[{native}]: '
                                  f'image={shape(img)}, image^={shape(img_aug)}, target={target}')

            return (img_aug, target, native, *rest)


def shape(image: Union[Img, torch.Tensor, np.ndarray]) -> str:
    if isinstance(image, Img):
        w, h = image.size
        prefix = 'image'
        dims = [h, w]
    elif isinstance(image, torch.Tensor):
        prefix = 'tensor'
        dims = list(image.shape)
    elif isinstance(image, np.ndarray):
        prefix = 'ndarray'
        dims = list(image.shape)
    else:
        raise NotImplementedError(f'image type: {type(image)}')

    dims = [str(dim) for dim in dims]
    return f'{prefix}({"x".join(dims)})'


def targets(dataset: Dataset) -> List[Any]:
    result = []

    for i in range(0, len(dataset)):
        _, target, *_ = dataset[i]
        result.append(target)

    return result
