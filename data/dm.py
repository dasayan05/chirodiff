import os
from enum import Enum
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning import LightningDataModule

from data.qd import (
    DS_threeseqdel,
    DS_threeseqabs,
    DS_threeseqdel_pointcloudcond,
    DS_threeseqdel_classcond,
    DS_threeseqabs_classcond,
    DS_threeseqabs_pointcloudcond,
    DS_threeseqabs_threeseqabscond,
    DS_threeseqdel_threeseqdelcond
)


class ReprType(str, Enum):
    threeseqdel = "threeseqdel"
    threeseqabs = "threeseqabs"
    threeseqabs_threeseqabscond = "threeseqabs_threeseqabscond"
    threeseqdel_pointcloudcond = "threeseqdel_pointcloudcond"
    threeseqdel_classcond = "threeseqdel_classcond"
    threeseqabs_classcond = "threeseqabs_classcond"
    threeseqabs_pointcloudcond = "threeseqabs_pointcloudcond"
    threeseqdel_threeseqdelcond = "threeseqdel_threeseqdelcond"


class GenericDM(LightningDataModule):

    def __init__(self, split_seed, split_fraction, batch_size, num_worker, repr):
        super().__init__()

        self.split_seed = split_seed
        self.split_fraction = split_fraction
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.repr = repr

        # subclasses need to set this with a 'Dataset' instance
        self._dataset = None

    @property
    def dataset(self):
        if self._dataset is None:
            raise ValueError(f'Subclass {self.__class__.__name__} is yet to assign a Dataset')
        else:
            return self._dataset

    @dataset.setter
    def dataset(self, d):
        if not isinstance(d, Dataset):
            raise ValueError(f'Expected a Dataset, got {d}')
        else:
            self._dataset = d

    def compute_split_size(self):
        self.train_len = int(len(self.dataset) * self.split_fraction)
        self.valid_len = len(self.dataset) - self.train_len

    def setup(self, stage: str):
        self.train_dataset, self.valid_dataset = \
            random_split(self.dataset, [self.train_len, self.valid_len],
                         torch.Generator().manual_seed(self.split_seed))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size, pin_memory=True, drop_last=True, shuffle=True,
                          num_workers=self.num_worker, collate_fn=self.dataset.__class__.collate)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size, pin_memory=True, drop_last=True, shuffle=True,
                          num_workers=self.num_worker, collate_fn=self.dataset.__class__.collate)

    def test_dataloader(self):
        return self.val_dataloader()


class QuickDrawDM(GenericDM):

    def __init__(self,
                 root_dir: str,
                 max_sketches: Optional[int] = None,
                 max_strokes: Optional[int] = None,
                 split_fraction: float = 0.85,
                 perlin_noise: float = 0.2,
                 penbit: bool = True,
                 split_seed: int = 4321,
                 batch_size: int = 64,
                 num_workers: int = os.cpu_count() // 2,
                 rdp: Optional[float] = None,
                 cond_rdp: Optional[float] = None,
                 repr: ReprType = ReprType.threeseqdel,
                 cache: bool = False
                 ):
        """QuickDraw Datamodule (OneSeq)

        Args:
            root_dir: Root directory of QD data (unpacked by `unpack_ndjson.py` utility)
            category: QD category name
            max_sketches: Maximum number of sketches to use
            max_strokes: clamp the maximum number of strokes (None for all strokes)
            split_fraction: Train/Validation split fraction
            perlin_noise: Strength of Perlin noise (YET TO BE IMPL)
            granularity: Number of points in each sample
            split_seed: Data splitting seed
            batch_size: Batch size for training
            rdp: RDP algorithm parameter ('None' to ignore RDP entirely)
            repr: data representation (oneseq or strokewise)
        """
        self.save_hyperparameters()
        self.hp = self.hparams  # an easier name
        super().__init__(self.hp.split_seed,
                         self.hp.split_fraction,
                         self.hp.batch_size,
                         self.hp.num_workers,
                         self.hp.repr)

        self._construct()

    def _construct(self):
        if self.hp.repr == ReprType.threeseqdel:
            self.dataset = DS_threeseqdel(self.hp.root_dir,
                                                   perlin_noise=self.hp.perlin_noise,
                                                   max_sketches=self.hp.max_sketches,
                                                   max_strokes=self.hp.max_strokes,
                                                   penbit=self.hp.penbit,
                                                   rdp=self.hp.rdp)
        elif self.hp.repr == ReprType.threeseqabs:
            self.dataset = DS_threeseqabs(self.hp.root_dir,
                                                   perlin_noise=self.hp.perlin_noise,
                                                   max_sketches=self.hp.max_sketches,
                                                   max_strokes=self.hp.max_strokes,
                                                   penbit=self.hp.penbit,
                                                   rdp=self.hp.rdp)
        elif self.hp.repr == ReprType.threeseqdel_pointcloudcond:
            self.dataset = DS_threeseqdel_pointcloudcond(self.hp.root_dir,
                                                   perlin_noise=self.hp.perlin_noise,
                                                   max_sketches=self.hp.max_sketches,
                                                   max_strokes=self.hp.max_strokes,
                                                   penbit=self.hp.penbit,
                                                   rdp=self.hp.rdp)
        elif self.hp.repr == ReprType.threeseqdel_classcond:
            self.dataset = DS_threeseqdel_classcond(self.hp.root_dir,
                                                   perlin_noise=self.hp.perlin_noise,
                                                   max_sketches=self.hp.max_sketches,
                                                   max_strokes=self.hp.max_strokes,
                                                   penbit=self.hp.penbit,
                                                   rdp=self.hp.rdp)
        elif self.hp.repr == ReprType.threeseqabs_classcond:
            self.dataset = DS_threeseqabs_classcond(self.hp.root_dir,
                                                   perlin_noise=self.hp.perlin_noise,
                                                   max_sketches=self.hp.max_sketches,
                                                   max_strokes=self.hp.max_strokes,
                                                   penbit=self.hp.penbit,
                                                   rdp=self.hp.rdp)
        elif self.hp.repr == ReprType.threeseqabs_pointcloudcond:
            self.dataset = DS_threeseqabs_pointcloudcond(self.hp.root_dir,
                                                   perlin_noise=self.hp.perlin_noise,
                                                   max_sketches=self.hp.max_sketches,
                                                   max_strokes=self.hp.max_strokes,
                                                   penbit=self.hp.penbit,
                                                   rdp=self.hp.rdp)
        elif self.hp.repr == ReprType.threeseqabs_threeseqabscond:
            self.dataset = DS_threeseqabs_threeseqabscond(self.hp.root_dir,
                                                   perlin_noise=self.hp.perlin_noise,
                                                   max_sketches=self.hp.max_sketches,
                                                   max_strokes=self.hp.max_strokes,
                                                   penbit=self.hp.penbit,
                                                   rdp=self.hp.rdp,
                                                   cond_rdp=self.hp.cond_rdp)
        elif self.hp.repr == ReprType.threeseqdel_threeseqdelcond:
            self.dataset = DS_threeseqdel_threeseqdelcond(self.hp.root_dir,
                                                   perlin_noise=self.hp.perlin_noise,
                                                   max_sketches=self.hp.max_sketches,
                                                   max_strokes=self.hp.max_strokes,
                                                   penbit=self.hp.penbit,
                                                   rdp=self.hp.rdp,
                                                   cond_rdp=self.hp.cond_rdp)
        else:
            pass

        self.compute_split_size()
