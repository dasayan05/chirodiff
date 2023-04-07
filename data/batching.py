import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy

from data.sketch import Sketch

# TODO: reasonable limit. can be made cmd arg later
MAX_SEQ_LENGTH = 300


class SketchRepr(object):

    def __init__(self, penbit=True, cache=False):
        super().__init__()
        
        self.penbit = penbit

    def represent(self, sketch):
        raise NotImplementedError('Abstract method not callable')

    def collate(self, batch: list):
        raise NotImplementedError('Abstract method not callable')


class Strokewise(SketchRepr):

    def __init__(self, penbit=True, cache=False):
        # Here 'granularity' means stroke-granularity
        super().__init__(penbit=penbit, cache=cache)

    def represent(self, sketch: Sketch):
        sk_repr = []

        total_seq_len = sum([len(stroke) for stroke in sketch])
        if total_seq_len >= MAX_SEQ_LENGTH:
            return None

        for stroke in sketch:
            seq_stroke, _ = stroke.tensorize()
            
            # TODO: clean this properly; strokes becomes (2,) sized
            if len(stroke) > 1 and len(stroke.stroke.shape) == 2:  # sloppy fix
                sk_repr.append({
                    'start': seq_stroke[0, :],
                    'time_range': torch.from_numpy(stroke.timestamps.astype(np.float32)),
                    'poly_stroke': seq_stroke
                })

        return sk_repr


class Pointcloud(Strokewise):

    def construct_sample(sk):
        if len(sk) == 0:
            return None

        sk_set = torch.cat([s['poly_stroke'] for s in sk], 0)
        return sk_set

    def represent(self, sketch: Sketch):
        sk = super().represent(sketch)
        return sk and Pointcloud.construct_sample(sk)

    def collate(batch: list):
        batch = [b for b in batch if b is not None]
        lens = torch.tensor([b.shape[0] for b in batch])
        pd = pad_sequence(batch, batch_first=True)
        return None, (pd, lens)


class ThreePointDelta(Strokewise):

    def construct_sample(sk, penbit=True):
        sk_list = []
        t_list = []
        for i, stroke in enumerate(sk):
            timestamps = stroke['time_range']
            stroke = stroke['poly_stroke']
            pen = torch.ones(stroke.shape[0], 1, dtype=stroke.dtype, device=stroke.device) * i
            pen[-1, 0] = i + 1
            sk_list.append(torch.cat([stroke, pen], -1))
            t_list.append(timestamps)
        
        if len(sk) == 0:
            return None

        sk = torch.cat(sk_list, 0)
        time = torch.cat(t_list, 0)
        sk_delta = sk[1:, :] - sk[:-1, :]
        if not penbit:
            sk_delta = sk_delta[:, :-1]

        time = time[:-1] # velocity is not available for the last point
        return torch.cat([sk_delta, time[:, None]], -1)

    def represent(self, sketch: Sketch):
        sk = super().represent(sketch)
        return sk and ThreePointDelta.construct_sample(sk, self.penbit)
    
    def collate(batch: list):
        batch = [b for b in batch if b is not None]
        lens = torch.tensor([b.shape[0] for b in batch])
        pd = pad_sequence(batch, batch_first=True)
        return None, (pd, lens)


class ThreePointDelta_PointCloudCond(Strokewise):

    def represent(self, sketch: Sketch):
        sk = super().represent(sketch)

        if sk is None:
            return None, None

        sk_threepointdelta = ThreePointDelta.construct_sample(sk, self.penbit)
        
        sketch = deepcopy(sketch)
        sk = super().represent(sketch)

        if sk is None:
            return None, None

        sk_pointcloud = Pointcloud.construct_sample(sk)

        return sk_pointcloud, sk_threepointdelta
    
    def collate(batch: list):
        sk_threepointdeltas = [tpd for _, tpd in batch]
        sk_pointclouds = [pc for pc, _ in batch]

        _, pc_batch = Pointcloud.collate(sk_pointclouds)
        _, tpd_batch = ThreePointDelta.collate(sk_threepointdeltas)
        return pc_batch, tpd_batch


class ThreePointAbs(Strokewise):

    def construct_sample(sk, penbit=True):
        sk_list = []
        t_list = []
        for _, stroke in enumerate(sk):
            timestamps = stroke['time_range']
            stroke = stroke['poly_stroke']
            pen = torch.zeros(stroke.shape[0], 1, dtype=stroke.dtype, device=stroke.device)
            pen[-1, 0] = 1.
            sk_list.append(torch.cat([stroke, pen], -1))
            t_list.append(timestamps)
        
        if len(sk) == 0:
            return None

        sk = torch.cat(sk_list, 0)
        if not penbit:
            sk = sk[:, :-1]

        time = torch.cat(t_list, 0)
        return torch.cat([sk[1:, :], time[1:, None]], -1)

    def represent(self, sketch: Sketch):
        sk = super().represent(sketch)
        return sk and ThreePointAbs.construct_sample(sk, self.penbit)
    
    def collate(batch: list):
        batch = [b for b in batch if b is not None]
        lens = torch.tensor([b.shape[0] for b in batch])
        pd = pad_sequence(batch, batch_first=True)
        return None, (pd, lens)


class ThreePointAbs_PointCloudCond(Strokewise):

    def represent(self, sketch: Sketch):
        sk = super().represent(sketch)

        if sk is None:
            return None, None

        sk_threepointabs = ThreePointAbs.construct_sample(sk, self.penbit)
        
        sketch = deepcopy(sketch)
        sk = super().represent(sketch)

        if sk is None:
            return None, None

        sk_pointcloud = Pointcloud.construct_sample(sk)

        return sk_pointcloud, sk_threepointabs
    
    def collate(batch: list):
        sk_threepointabss = [tpa for _, tpa in batch]
        sk_pointclouds = [pc for pc, _ in batch]

        _, pc_batch = Pointcloud.collate(sk_pointclouds)
        _, tpa_batch = ThreePointAbs.collate(sk_threepointabss)
        return pc_batch, tpa_batch


class ThreePointAbs_ThreeSeqAbs(Strokewise):

    def __init__(self, penbit=True, cond_rdp=None, cache=False):
        super().__init__(penbit, cache)

        self.cond_rdp = cond_rdp

    def represent(self, sketch: Sketch):
        cond_sketch = deepcopy(sketch)

        # spatially scaling back to 1. and then 10. is needed because the stuff in the middle
        # (resampling rate, RDP parameter) are sensitive to spatial scale of the vector entity.
        cond_sketch.scale_spatial(1.)
        if self.cond_rdp is not None:
            cond_sketch.rdp(self.cond_rdp)
        cond_sketch.scale_spatial(10.)

        cond_sk = super().represent(cond_sketch)
        sk = super().represent(sketch)

        if sk is None or cond_sk is None:
            return None, None
        
        cond_sk_threepointabs = ThreePointAbs.construct_sample(cond_sk, self.penbit)
        sk_threepointabs = ThreePointAbs.construct_sample(sk, self.penbit)

        # timestep not needed for the condition
        return cond_sk_threepointabs, \
            sk_threepointabs

    def collate(batch: list):
        sk_threepointabss = [h_tpa for _, h_tpa in batch]
        cond_sk_threepointabss = [l_tpa for l_tpa, _ in batch]

        _, tpa_batch = ThreePointAbs.collate(sk_threepointabss)
        _, cond_tpa_batch = ThreePointAbs.collate(cond_sk_threepointabss)
        return cond_tpa_batch, tpa_batch


class ThreePointDel_ThreeSeqDel(Strokewise):

    def __init__(self, penbit=True, cond_rdp=None, cache=False):
        super().__init__(penbit, cache)

        self.cond_rdp = cond_rdp

    def represent(self, sketch: Sketch):
        cond_sketch = deepcopy(sketch)

        # spatially scaling back to 1. and then 10. is needed because the stuff in the middle
        # (resampling rate, RDP parameter) are sensitive to spatial scale of the vector entity.
        cond_sketch.scale_spatial(1.)
        if self.cond_rdp is not None:
            cond_sketch.rdp(self.cond_rdp)
        cond_sketch.scale_spatial(10.)

        cond_sk = super().represent(cond_sketch)
        sk = super().represent(sketch)

        if sk is None or cond_sk is None:
            return None, None
        
        cond_sk_threepointdel = ThreePointDelta.construct_sample(cond_sk, self.penbit)
        sk_threepointdel = ThreePointDelta.construct_sample(sk, self.penbit)

        # timestep not needed for the condition
        return cond_sk_threepointdel, \
            sk_threepointdel

    def collate(batch: list):
        sk_threepointdels = [h_tpd for _, h_tpd in batch]
        cond_sk_threepointdels = [l_tpd for l_tpd, _ in batch]

        _, tpd_batch = ThreePointAbs.collate(sk_threepointdels)
        _, cond_tpd_batch = ThreePointAbs.collate(cond_sk_threepointdels)
        return cond_tpd_batch, tpd_batch


class StrokeSet(Strokewise):

    def represent(self, sketch: Sketch):
        sk = super().represent(sketch)

        sk_list = []
        for stroke in sk:
            abs_stroke = stroke['poly_stroke']
            del_stroke = abs_stroke[1:, ...] - abs_stroke[:-1, ...]
            start_del_stroke = torch.cat([stroke['start'][None, :], del_stroke], 0)
            sk_list.append(start_del_stroke.ravel())
        
        sk = torch.stack(sk_list, 0)
        return sk

    def collate(batch: list):
        batch = [b for b in batch if b is not None]
        lens = torch.tensor([b.shape[0] for b in batch])
        pd = pad_sequence(batch, batch_first=True)
        return pd, lens
