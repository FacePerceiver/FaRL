# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import cv2

from blueprint.ml import Dataset, Split


class LaPa(Dataset):
    """LaPa face parsing dataset

    Args:
        root (str): The directory that contains subdirs 'image', 'labels'
    """

    def __init__(self, root, split=Split.ALL):
        assert os.path.isdir(root)
        self.root = root

        subfolders = []
        if split == Split.TRAIN:
            subfolders = ['train']
        elif split == Split.VAL:
            subfolders = ['val']
        elif split in {Split.TEST, Split.TOY}:
            subfolders = ['test']
        elif split == Split.ALL:
            subfolders = ['train', 'val', 'test']

        self.info = []
        for subf in subfolders:
            for name in os.listdir(os.path.join(self.root, subf, 'images')):
                if not name.endswith('.jpg'):
                    continue
                name = name.split('.')[0]
                image_path = os.path.join(
                    self.root, subf, 'images', f'{name}.jpg')
                label_path = os.path.join(
                    self.root, subf, 'labels', f'{name}.png')
                landmark_path = os.path.join(
                    self.root, subf, 'landmarks', f'{name}.txt')
                assert os.path.exists(image_path)
                assert os.path.exists(label_path)
                assert os.path.exists(landmark_path)
                landmarks = [float(v) for v in open(
                    landmark_path, 'r').read().split()]
                assert landmarks[0] == 106 and len(landmarks) == 106*2+1
                landmarks = np.reshape(
                    np.array(landmarks[1:], np.float32), [106, 2])
                sample_name = f'{subf}.{name}'
                self.info.append(
                    {'image_path': image_path, 'label_path': label_path,
                     'landmarks': landmarks, 'sample_name': sample_name})
                if split == Split.TOY and len(self.info) >= 10:
                    break

    def __getitem__(self, index):
        info = self.info[index]
        image = cv2.imread(info['image_path'])[:, :, ::-1]
        label = cv2.imread(info['label_path'], cv2.IMREAD_GRAYSCALE)
        landmarks = info['landmarks']
        return {'image': image, 'label': label, 'landmarks': landmarks}

    def __len__(self):
        return len(self.info)

    def sample_name(self, index):
        return self.info[index]['sample_name']

    @property
    def label_names(self):
        return ['background', 'face_lr_rr', 'lb', 'rb', 'le', 're', 'nose', 'ul', 'im', 'll', 'hair']

    @staticmethod
    def draw_landmarks(im, landmarks, color, thickness=5, eye_radius=3):
        landmarks = landmarks.astype(np.int32)
        cv2.polylines(im, [landmarks[0:33]], False,
                      color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[33:42]], True,
                      color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[42:51]], True,
                      color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[51:55]], False,
                      color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[55:66]], False,
                      color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[66:74]], True,
                      color, thickness, cv2.LINE_AA)
        cv2.circle(im, (landmarks[74, 0], landmarks[74, 1]),
                   eye_radius, color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[75:83]], True,
                      color, thickness, cv2.LINE_AA)
        cv2.circle(im, (landmarks[83, 0], landmarks[83, 1]),
                   eye_radius, color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[84:96]], True,
                      color, thickness, cv2.LINE_AA)
        cv2.polylines(im, [landmarks[96:-2]], True,
                      color, thickness, cv2.LINE_AA)
        return im
