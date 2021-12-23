# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import Optional
import numpy as np
import cv2

from blueprint.ml import Dataset, Split


class IBUG300W(Dataset):
    def __init__(self, root: str, split: Split, subset: Optional[str] = None):
        self.root = root
        self.anno = []

        if split == Split.TRAIN:
            anno_file = 'face_landmarks_300w_train.csv'
        elif split == Split.TEST:
            if subset == 'Common':
                anno_file = 'face_landmarks_300w_valid_common.csv'
            elif subset == 'Challenging':
                anno_file = 'face_landmarks_300w_valid_challenge.csv'
            else:
                raise RuntimeError(
                    f'Invalid subset {subset} for IBUG300W test set (should be "Common" or "Challenging")')
        else:
            raise RuntimeError(f'Unsupported split {split} for IBUG300W')

        error_im_paths = {
            'ibug/image_092_01.jpg': 'ibug/image_092 _01.jpg'
        }

        self.info_list = []
        with open(os.path.join(self.root, anno_file), 'r') as fd:
            fd.readline()  # skip the first line
            for line in fd:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.startswith('#'):
                    continue
                im_path, scale, center_w, center_h, * \
                    landmarks = line.split(',')

                if im_path in error_im_paths:
                    im_path = error_im_paths[im_path]

                sample_name = os.path.splitext(im_path)[0].replace('/', '_')

                im_path = os.path.join(self.root, im_path)
                assert os.path.exists(im_path)

                self.info_list.append({
                    'sample_name': sample_name,
                    'im_path': im_path,
                    'landmarks': np.reshape(np.array([float(v)-2.0 for v in landmarks], dtype=np.float32), [68, 2]),
                    'box_info': (float(scale), float(center_w)-2.0, float(center_h)-2.0)
                })

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, index):
        info = self.info_list[index]
        image = cv2.cvtColor(cv2.imread(info['im_path']), cv2.COLOR_BGR2RGB)
        scale, center_w, center_h = info['box_info']
        box_half_size = 100.0 * scale

        return {
            'image': image,
            'box': np.array([center_h-box_half_size, center_w-box_half_size,
                             center_h+box_half_size, center_w+box_half_size],
                            dtype=np.float32),
            'landmarks': info['landmarks']
        }

    def sample_name(self, index):
        return self.info_list[index]['sample_name']
