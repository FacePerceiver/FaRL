import os
import numpy as np
import cv2
import functools
from typing import Dict, List

from blueprint.ml import Dataset, Split


@functools.lru_cache()
def _cached_imread(fname, flags=None):
    return cv2.imread(fname, flags=flags)


class CelebAMaskHQ(Dataset):
    def __init__(self, root, split, label_type='all'):
        assert os.path.isdir(root)
        self.root = root
        self.split = split
        self.names = []

        if split != Split.ALL:
            hq_to_orig_mapping = dict()
            orig_to_hq_mapping = dict()
            mapping_file = os.path.join(
                root, 'CelebA-HQ-to-CelebA-mapping.txt')
            assert os.path.exists(mapping_file)
            for s in open(mapping_file, 'r'):
                if '.jpg' not in s:
                    continue
                idx, _, orig_file = s.split()
                hq_to_orig_mapping[int(idx)] = orig_file
                orig_to_hq_mapping[orig_file] = int(idx)

            # load partition
            partition_file = os.path.join(root, 'list_eval_partition.txt')
            assert os.path.exists(partition_file)
            for s in open(partition_file, 'r'):
                if '.jpg' not in s:
                    continue
                orig_file, group = s.split()
                group = int(group)
                if orig_file not in orig_to_hq_mapping:
                    continue
                hq_id = orig_to_hq_mapping[orig_file]
                if split == Split.TRAIN and group == 0:
                    self.names.append(str(hq_id))
                elif split == Split.VAL and group == 1:
                    self.names.append(str(hq_id))
                elif split == Split.TEST and group == 2:
                    self.names.append(str(hq_id))
                elif split == Split.TOY:
                    self.names.append(str(hq_id))
                    if len(self.names) >= 10:
                        break
        else:
            self.names = [
                n[:-(len('.jpg'))]
                for n in os.listdir(os.path.join(self.root, 'CelebA-HQ-img'))
                if n.endswith('.jpg')
            ]

        self.label_setting = {
            'human': {
                'suffix': [
                    'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow',
                    'l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair'
                ],
                'names': [
                    'bg', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're',
                    'le', 'nose', 'imouth', 'llip', 'ulip', 'hair'
                ]
            },
            'aux': {
                'suffix': [
                    'eye_g', 'hat', 'ear_r', 'neck_l',
                ],
                'names': [
                    'normal', 'glass', 'hat', 'earr', 'neckl'
                ]
            },
            'all': {
                'suffix': [
                    'neck', 'skin', 'cloth', 'l_ear', 'r_ear', 'l_brow', 'r_brow',
                    'l_eye', 'r_eye', 'nose', 'mouth', 'l_lip', 'u_lip', 'hair',
                    'eye_g', 'hat', 'ear_r', 'neck_l',
                ],
                'names': [
                    'bg', 'neck', 'face', 'cloth', 'rr', 'lr', 'rb', 'lb', 're',
                    'le', 'nose', 'imouth', 'llip', 'ulip', 'hair',
                    'glass', 'hat', 'earr', 'neckl'
                ]
            }
        }[label_type]

    def make_label(self, index, ordered_label_suffix):
        label = np.zeros((512, 512), np.uint8)
        name = self.names[index]
        name_id = int(name)
        name5 = '%05d' % name_id
        p = os.path.join(self.root, 'CelebAMask-HQ-mask-anno',
                         str(name_id // 2000), name5)
        for i, label_suffix in enumerate(ordered_label_suffix):
            label_value = i + 1
            label_fname = os.path.join(p + '_' + label_suffix + '.png')
            if os.path.exists(label_fname):
                mask = _cached_imread(label_fname, cv2.IMREAD_GRAYSCALE)
                label = np.where(mask > 0,
                                 np.ones_like(label) * label_value, label)
        return label

    def __getitem__(self, index):
        name = self.names[index]
        image = cv2.resize(
            cv2.imread(os.path.join(self.root, 'CelebA-HQ-img',
                                    name + '.jpg'))[:, :, ::-1],
            (512, 512),
            interpolation=cv2.INTER_LINEAR)

        data = {'image': image}
        label = self.make_label(index, self.label_setting['suffix'])
        data[f'label'] = label

        return data

    def __len__(self):
        return len(self.names)

    def sample_name(self, index):
        return self.names[index]

    @property
    def label_names(self) -> List[str]:
        return self.label_setting['names']
