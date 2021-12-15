from typing import Mapping, List, Union, Optional
from collections import defaultdict

import math

import numpy as np
from scipy.integrate import simps
import torch
import torch.distributed as dist

from blueprint.ml import Scorer, all_gather_by_part


class NormalizeInfo:
    def get_unit_dist(self, data) -> float:
        raise NotImplementedError()


class NormalizeByLandmarks(NormalizeInfo):
    def __init__(self, landmark_tag: str, left_id: Union[int, List[int]], right_id: Union[int, List[int]]):
        self.landmark_tag = landmark_tag
        if isinstance(left_id, int):
            left_id = [left_id]
        if isinstance(right_id, int):
            right_id = [right_id]
        self.left_id, self.right_id = left_id, right_id

    def get_unit_dist(self, data) -> float:
        landmark = data[self.landmark_tag]
        unit_dist = np.linalg.norm(landmark[self.left_id, :].mean(0) -
                                   landmark[self.right_id, :].mean(0), axis=-1)
        return unit_dist


class NormalizeByBox(NormalizeInfo):
    def __init__(self, box_tag: str):
        self.box_tag = box_tag

    def get_unit_dist(self, data) -> float:
        y1, x1, y2, x2 = data[self.box_tag]
        h = y2 - y1
        w = x2 - x1
        return math.sqrt(h * w)


class NormalizeByBoxDiag(NormalizeInfo):
    def __init__(self, box_tag: str):
        self.box_tag = box_tag

    def get_unit_dist(self, data) -> float:
        y1, x1, y2, x2 = data[self.box_tag]
        h = y2 - y1
        w = x2 - x1
        diag = math.sqrt(w * w + h * h)
        return diag


class NME(Scorer):
    """Compute Normalized Mean Error among 2D landmarks and predicted 2D landmarks.

    Attributes:
        normalize_infos: Mapping[str, NormalizeInfo]: 
            Information to normalize for NME calculation.
    """

    def __init__(self, landmark_tag: str, pred_landmark_tag: str,
                 normalize_infos: Mapping[str, NormalizeInfo]) -> None:
        self.landmark_tag = landmark_tag
        self.pred_landmark_tag = pred_landmark_tag
        self.normalize_infos = normalize_infos

    def init_evaluation(self):
        self.nmes_sum = defaultdict(float)  # norm_name: str -> float
        self.count = defaultdict(int)  # norm_name: str -> int

    def evaluate(self, data: Mapping[str, np.ndarray]):
        landmark = data[self.landmark_tag]
        pred_landmark = data[self.pred_landmark_tag]

        if landmark.shape != pred_landmark.shape:
            raise RuntimeError(
                f'The landmark shape {landmark.shape} mismatches '
                f'the pred_landmark shape {pred_landmark.shape}')

        for norm_name, norm_info in self.normalize_infos.items():
            # compute unit distance for nme normalization
            unit_dist = norm_info.get_unit_dist(data)

            # compute normalized nme for this sample
            # [npoints] -> scalar
            nme = (np.linalg.norm(
                landmark - pred_landmark, axis=-1) / unit_dist).mean()
            self.nmes_sum[norm_name] += nme

            self.count[norm_name] += 1

    def finalize_evaluation(self) -> Mapping[str, float]:
        # gather all nmes_sum
        names_array: List[str] = list(self.nmes_sum.keys())

        nmes_sum = torch.tensor(
            [self.nmes_sum[name] for name in names_array],
            dtype=torch.float32, device='cuda')
        if dist.is_initialized():
            dist.all_reduce(nmes_sum)

        count_sum = torch.tensor(
            [self.count[name] for name in names_array],
            dtype=torch.int64, device='cuda')
        if dist.is_initialized():
            dist.all_reduce(count_sum)

        scores = dict()

        # compute nme scores
        for name, nmes_sum_val, count_val in zip(names_array, nmes_sum, count_sum):
            # print(f'Note: NME is calculated with '
            #       f'{count_val.item()} data in total')
            scores[name] = nmes_sum_val.item() / count_val.item()

        # compute final nme
        return scores


class AUC_FR(Scorer):
    """Compute AUC and FR (Failure Rate).

    Output scores with name `'auc_{suffix_name}'` and `'fr_{suffix_name}'`.
    """

    def __init__(self, landmark_tag: str, pred_landmark_tag: str,
                 normalize_info: NormalizeInfo,
                 threshold: float, suffix_name: str, step: float = 0.0001,
                 gather_part_size: Optional[int] = 5) -> None:
        self.landmark_tag = landmark_tag
        self.pred_landmark_tag = pred_landmark_tag
        self.normalize_info = normalize_info
        self.threshold = threshold
        self.suffix_name = suffix_name
        self.step = step
        self.gather_part_size = gather_part_size

    def init_evaluation(self):
        self.nmes = []

    def evaluate(self, data: Mapping[str, np.ndarray]):
        landmark = data[self.landmark_tag]
        pred_landmark = data[self.pred_landmark_tag]

        if landmark.shape != pred_landmark.shape:
            raise RuntimeError(
                f'The landmark shape {landmark.shape} mismatches '
                f'the pred_landmark shape {pred_landmark.shape}')

        # compute unit distance for nme normalization
        unit_dist = self.normalize_info.get_unit_dist(data)

        # compute normalized nme for this sample
        nme = (np.linalg.norm(
            landmark - pred_landmark, axis=-1) / unit_dist).mean()
        self.nmes.append(nme)

    def finalize_evaluation(self) -> Mapping[str, float]:
        # gather all nmes

        if dist.is_initialized():
            nmes = all_gather_by_part(self.nmes, self.gather_part_size)
        else:
            nmes = self.nmes
        nmes = torch.tensor(nmes)

        # nmes = torch.tensor(self.nmes, dtype=torch.float32, device='cuda')
        # if dist.is_initialized():
        #     all_nmes_list = [t.cuda() for t in all_gather(nmes)]  # list of tensors
        #     nmes = torch.cat(all_nmes_list)
        # print(f'In total {nmes.size(0)} nmes are collected.')

        nmes = nmes.sort(dim=0).values.cpu().numpy()

        # from https://github.com/HRNet/HRNet-Facial-Landmark-Detection/issues/6#issuecomment-503898737
        count = len(nmes)
        xaxis = list(np.arange(0., self.threshold + self.step, self.step))
        ced = [float(np.count_nonzero([nmes <= x])) / count for x in xaxis]
        auc = simps(ced, x=xaxis) / self.threshold
        fr = 1. - ced[-1]

        # # compute ced locally
        # xaxis = list(np.arange(0., self.threshold + self.step, self.step))
        # local_count = len(self.nmes)
        # local_ced_raw = np.array(
        #     [float(np.count_nonzero([self.nmes <= x])) for x in xaxis])
        # if dist.is_initialized():
        #     placeholder = torch.tensor(
        #         local_count, dtype=torch.int32, device='cuda')
        #     dist.reduce(placeholder)
        #     count = placeholder.item()

        #     placeholder = torch.from_numpy(local_ced_raw).to(
        #         dtype=torch.int32, device='cuda')
        #     dist.reduce(placeholder)
        #     ced_raw = placeholder.cpu().numpy()
        # else:
        #     count = local_count
        #     ced_raw = local_ced_raw

        # ced = ced_raw / float(count)
        # auc = simps(ced, x=xaxis) / self.threshold
        # fr = 1. - ced[-1]

        return {f'auc_{self.suffix_name}': auc, f'fr_{self.suffix_name}': fr}
