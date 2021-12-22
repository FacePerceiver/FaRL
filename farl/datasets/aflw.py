# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import cv2
import scipy.io

from blueprint.ml import Dataset, Split


class AFLW_19(Dataset):
    def __init__(self, root, split=Split.ALL, subset: str = 'full'):
        self.images_root = os.path.join(root, 'data', 'flickr')
        info = scipy.io.loadmat(os.path.join(
            root, 'AFLWinfo_release.mat'))
        self.bbox = info['bbox']  # 24386x4 left, right, top bottom
        self.data = info['data']  # 24386x38 x1,x2...,xn,y1,y2...,yn
        self.mask = info['mask_new']  # 24386x19
        self.name_list = [s[0][0] for s in info['nameList']]

        ra = np.reshape(info['ra'].astype(np.int32), [-1])-1
        assert ra.min() == 0
        assert ra.max() == self.bbox.shape[0] - 1
        if split == Split.ALL:
            self.indices = ra
        elif split == Split.TRAIN:
            self.indices = ra[:20000]
        elif split == Split.TEST:
            if subset == 'full':
                self.indices = ra[20000:]
            elif subset == 'frontal':
                all_visible = np.all(self.mask == 1, axis=1)  # 24386
                self.indices = np.array(
                    [ind for ind in ra[20000:] if all_visible[ind]])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        ind = self.indices[index]
        image_path = os.path.join(
            self.images_root, self.name_list[ind])
        assert os.path.exists(image_path)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        landmarks = np.reshape(self.data[ind], [2, 19]).transpose()

        left, right, top, bottom = self.bbox[ind]
        box_y1x1y2x2 = np.array([top, left, bottom, right], dtype=np.float32)

        visibility = self.mask[ind]
        return {
            'image': image,
            'box': box_y1x1y2x2,
            'landmarks': landmarks,
            'visibility': visibility
        }

    def sample_name(self, index):
        return str(index)
