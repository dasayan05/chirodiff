import os
import sys
import random
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from data.sketch import Sketch
from data.batching import Pointcloud, Strokewise, \
    ThreePointAbs, ThreePointDelta, ThreePointDelta_PointCloudCond, \
        ThreePointAbs_PointCloudCond, ThreePointAbs_ThreeSeqAbs, \
            ThreePointDel_ThreeSeqDel


class QuickDraw(Dataset):

    def __init__(self, data_root, shuffle=True, perlin_noise=0.2,
                max_sketches=10000, max_strokes=None, rdp=None, **kwargs):
        super().__init__()

        self.data_root = data_root

        if os.path.isfile(self.data_root) and self.data_root.endswith('.npz'):
            self.npz_ptr = np.load(self.data_root, allow_pickle=True)
            self.attrs = self.npz_ptr.files
            self.data = {attr: self.npz_ptr[attr] for attr in self.attrs}
            self.cached = True
            return
        else:
            self.cached = False

        self.max_sketches = max_sketches
        self.max_strokes = max_strokes
        self.perlin_noise = perlin_noise
        self.rdp = rdp

        self.content_list = os.listdir(self.data_root)
        if all([os.path.isdir(os.path.join(self.data_root, c_path)) for c_path in self.content_list]):
            self.categories = self.content_list
            self.n_categories = len(self.categories)
            self.file_list = []
            for cat in self.categories:
                cat_content = os.listdir(os.path.join(self.data_root, cat))
                
                if self.max_sketches is not None:
                    max_sketches_per_cat = min(self.max_sketches, len(cat_content))
                    del cat_content[max_sketches_per_cat:]
                
                self.file_list.extend([os.path.join(cat, c) for c in cat_content])

        else:
            self.categories = None
            self.file_list = self.content_list

            if self.max_sketches is not None:
                max_sketches = min(self.max_sketches, len(self.file_list))
                del self.file_list[max_sketches:]

        if shuffle:
            random.shuffle(self.file_list)


    def __len__(self):
        if not self.cached:
            return len(self.file_list)
        else:
            return self.data[self.attrs[0]].shape[0]

    def get_sketch(self, i):
        if self.categories is not None:
            cat, _ = self.file_list[i].split('/')
            assert cat in self.categories, "something wrong with category/folder names"
            self.cat_idx = self.categories.index(cat)
        else:
            self.cat_idx = None

        file_path = os.path.join(self.data_root, self.file_list[i])

        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

            stroke_list = self.data['drawing']

            if self.max_strokes is not None:
                max_strokes = min(self.max_strokes, len(stroke_list))
                stroke_list = stroke_list[:max_strokes]

            sketch = Sketch(stroke_list, label=self.cat_idx)

            seed = random.randint(0, 10000)
            sketch.jitter(seed=seed, noise_level=self.perlin_noise)

            sketch.move()
            sketch.shift_time(0)
            sketch.scale_spatial(1)
            sketch.resample(delta=0.05)
            if self.rdp is not None:
                sketch.rdp(eps=self.rdp)
            sketch.scale_spatial(10)
            sketch.scale_time(1)

        return sketch

    def __getitem__(self, i):
        if not self.cached:
            return self.represent(self.get_sketch(i))
        else:
            if len(self.attrs) > 1:
                return tuple(torch.from_numpy(self.data[attr][i]) for attr in self.attrs)
            else:
                return torch.from_numpy(self.data[self.attrs[0]][i])


class QDSketchStrokewise(QuickDraw, Strokewise):

    def __init__(self, *args, **kwargs):
        QuickDraw.__init__(self, *args, **kwargs)
        Strokewise.__init__(self)

    def __getitem__(self, i):
        return self.represent(super().get_sketch(i))


class QDSketchPointcloud(QuickDraw, Pointcloud):

    def __init__(self, *args, **kwargs):
        QuickDraw.__init__(self, *args, **kwargs)
        Pointcloud.__init__(self)

    def __getitem__(self, i):
        return self.represent(super().__getitem__(i))


class DS_threeseqdel(QuickDraw, ThreePointDelta):

    def __init__(self, *args, **kwargs):
        QuickDraw.__init__(self, *args, **kwargs)
        ThreePointDelta.__init__(self, penbit=kwargs.get('penbit', True))
        

class DS_threeseqabs(QuickDraw, ThreePointAbs):

    def __init__(self, *args, **kwargs):
        QuickDraw.__init__(self, *args, **kwargs)
        ThreePointAbs.__init__(self, penbit=kwargs.get('penbit', True))


class DS_threeseqabs_classcond(QuickDraw, ThreePointAbs):

    def __init__(self, *args, **kwargs):
        QuickDraw.__init__(self, *args, **kwargs)
        ThreePointAbs.__init__(self, penbit=kwargs.get('penbit', True))
    
    def represent(self, sketch: Sketch):
        label = torch.tensor(sketch.label, dtype=torch.int64)
        return label, super().represent(sketch)
    
    def collate(batch: list):
        class_batch = torch.stack([c for c, _ in batch], 0)
        _, tpa_batch = ThreePointAbs.collate([tpa for _, tpa in batch])
        return class_batch, tpa_batch


class DS_threeseqdel_classcond(QuickDraw, ThreePointDelta):

    def __init__(self, *args, **kwargs):
        QuickDraw.__init__(self, *args, **kwargs)
        ThreePointDelta.__init__(self, penbit=kwargs.get('penbit', True))
    
    def represent(self, sketch: Sketch):
        label = torch.tensor(sketch.label, dtype=torch.int64)
        return label, super().represent(sketch)
    
    def collate(batch: list):
        class_batch = torch.stack([c for c, _ in batch], 0)
        _, tpd_batch = ThreePointDelta.collate([tpd for _, tpd in batch])
        return class_batch, tpd_batch


class DS_threeseqdel_pointcloudcond(QuickDraw, ThreePointDelta_PointCloudCond):

    def __init__(self, *args, **kwargs):
        QuickDraw.__init__(self, *args, **kwargs)
        ThreePointDelta_PointCloudCond.__init__(self, penbit=kwargs.get('penbit', True))


class DS_threeseqabs_pointcloudcond(QuickDraw, ThreePointAbs_PointCloudCond):

    def __init__(self, *args, **kwargs):
        QuickDraw.__init__(self, *args, **kwargs)
        ThreePointAbs_PointCloudCond.__init__(self, penbit=kwargs.get('penbit', True))


class DS_threeseqabs_threeseqabscond(QuickDraw, ThreePointAbs_ThreeSeqAbs):

    def __init__(self, *args, **kwargs):
        QuickDraw.__init__(self, *args, **kwargs)
        ThreePointAbs_ThreeSeqAbs.__init__(self, penbit=kwargs.get('penbit', True),
            cond_rdp=kwargs.get('cond_rdp', None))

    def __getitem__(self, i):
        if not self.cached:
            return self.represent(self.get_sketch(i))
        else:
            if len(self.attrs) > 1:
                return tuple(torch.from_numpy(self.data[attr][i]) for attr in self.attrs)
            else:
                # in case we need the same data as cond
                d = self.data[self.attrs[0]][i]
                return torch.from_numpy(d), torch.from_numpy(d)


class DS_threeseqdel_threeseqdelcond(QuickDraw, ThreePointDel_ThreeSeqDel):

    def __init__(self, *args, **kwargs):
        QuickDraw.__init__(self, *args, **kwargs)
        ThreePointDel_ThreeSeqDel.__init__(self, penbit=kwargs.get('penbit', True),
            cond_rdp=kwargs.get('cond_rdp', None))

    def __getitem__(self, i):
        if not self.cached:
            return self.represent(self.get_sketch(i))
        else:
            if len(self.attrs) > 1:
                return tuple(torch.from_numpy(self.data[attr][i]) for attr in self.attrs)
            else:
                # in case we need the same data as cond
                d = self.data[self.attrs[0]][i]
                return torch.from_numpy(d), torch.from_numpy(d)


if __name__ == '__main__':
    class_name_str = eval('DS_' + sys.argv[2])
    ds = class_name_str(
        sys.argv[1],
        perlin_noise=0.,
        max_sketches=100000,
        max_strokes=25,
        penbit=True,
        rdp=None
    )
    dummy_sample = ds[0]
    if not isinstance(dummy_sample, tuple):
        n_attr = 1
    else:
        n_attr = len(dummy_sample)

    samples = [[] for _ in range(n_attr)]
    for sam in tqdm(ds):
        if n_attr == 1:
            sam = (sam, )
        for a in range(n_attr):
            if sam[a] is None:
                break
            samples[a].append(sam[a].numpy())
    samples = [np.array(sams, dtype=np.ndarray) for sams in samples]
    attrs = [f'attr{a}' for a in range(n_attr)]
    
    np.savez(sys.argv[1] + f'_{sys.argv[2]}.npz', **dict(zip(attrs, samples)))