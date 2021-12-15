# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import argparse

from blueprint.ml import TRAIN, TEST, freeze, storage
from blueprint.ml import augmenters as aug

from .lapa import LaPa
from .celebamask_hq import CelebAMaskHQ
from .aflw import AFLW_19
from .ibug300w import IBUG300W
from .wflw import WFLW


def freeze_lapa(data_root):
    print('Freezing LaPa ...')
    ds_train = LaPa(os.path.join(data_root, 'LaPa'), TRAIN)
    ds_test = LaPa(os.path.join(data_root, 'LaPa'), TEST)
    print(f'train: {len(ds_train)}, test: {len(ds_test)}')

    aug_512 = [
        aug.With('image', 'image', aug.Normalize255()),
        aug.With('landmarks', 'face_align_pts',
                 lambda landmarks:landmarks[[104, 105, 54, 84, 90], :]),
        aug.With('face_align_pts', 'align_matrix',
                 aug.GetFaceAlignMatrix(target_shape=(512, 512))),
        aug.Filter(['image', 'label', 'align_matrix'])
    ]

    ds_train_aug = ds_train.augment(aug_512)
    ds_test_aug = ds_test.augment(aug_512)

    freeze(ds_test_aug, os.path.join(data_root, 'LaPa.test.zip'),
           {'image': storage.IMAGE_JPG,
            'label': storage.IMAGE_LABEL}, with_prog=True)
    freeze(ds_train_aug, os.path.join(data_root, 'LaPa.train.zip'),
           {'image': storage.IMAGE_JPG,
            'label': storage.IMAGE_LABEL}, with_prog=True)


def freeze_celebamaskhq(data_root):
    print('Freezing CelebAMaskHQ ...')
    ds_train = CelebAMaskHQ(os.path.join(data_root, 'CelebAMaskHQ'), TRAIN)
    ds_test = CelebAMaskHQ(os.path.join(data_root, 'CelebAMaskHQ'), TEST)
    print(f'train: {len(ds_train)}, test: {len(ds_test)}')

    aug_512 = [
        aug.AttachConstData('align_matrix', np.eye(3, dtype=np.float32)),
        aug.Filter(['image', 'label', 'align_matrix'])
    ]

    ds_train_aug = ds_train.augment(aug_512)
    ds_test_aug = ds_test.augment(aug_512)

    freeze(ds_test_aug, os.path.join(data_root, 'CelebAMaskHQ.test.zip'),
           {'image': storage.IMAGE_JPG,
            'label': storage.IMAGE_LABEL}, with_prog=True)
    freeze(ds_train_aug, os.path.join(data_root, 'CelebAMaskHQ.train.zip'),
           {'image': storage.IMAGE_JPG,
            'label': storage.IMAGE_LABEL}, with_prog=True)


def freeze_aflw19(data_root):
    print('Freezing AFLW19 ...')
    ds_train = AFLW_19(os.path.join(data_root, 'AFLW-19'), split=TRAIN)
    ds_test = AFLW_19(os.path.join(data_root, 'AFLW-19'), split=TEST)
    ds_test_frontal = AFLW_19(os.path.join(
        data_root, 'AFLW-19'), split=TEST, subset='frontal')

    print(f'train: {len(ds_train)}, test: {len(ds_test)}, '
          f'test_frontal: {len(ds_test_frontal)}')

    aug_512 = [
        aug.With(('box', None), 'crop_matrix', aug.UpdateCropAndResizeMatrix(
            (512, 512), align_corners=False)),
        aug.Filter(['image', 'label', 'landmarks', 'crop_matrix', 'box'])
    ]

    ds_train_aug = ds_train.augment(aug_512)
    ds_test_aug = ds_test.augment(aug_512)
    ds_test_frontal_aug = ds_test_frontal.augment(aug_512)

    freeze(ds_test_aug, os.path.join(data_root, 'AFLW-19.test.zip'),
           {'image': storage.IMAGE_JPG}, with_prog=True)
    freeze(ds_train_aug, os.path.join(data_root, 'AFLW-19.train.zip'),
           {'image': storage.IMAGE_JPG}, with_prog=True)
    freeze(ds_test_frontal_aug, os.path.join(data_root, 'AFLW-19.test_frontal.zip'),
           {'image': storage.IMAGE_JPG}, with_prog=True)


def freeze_ibug300w(data_root):
    print('Freezing IBUG300W ...')
    ds_train = IBUG300W(os.path.join(data_root, 'IBUG300W'), split=TRAIN)
    ds_test_common = IBUG300W(os.path.join(
        data_root, 'IBUG300W'), split=TEST, subset='Common')
    ds_test_challenging = IBUG300W(os.path.join(
        data_root, 'IBUG300W'), split=TEST, subset='Challenging')

    print(f'train: {len(ds_train)}, test_common: {len(ds_test_common)}, '
          f'test_challenging: {len(ds_test_challenging)}')

    aug_512 = [
        aug.With(('box', None), 'crop_matrix', aug.UpdateCropAndResizeMatrix(
            (512, 512), align_corners=False)),
        aug.Filter(['image', 'label', 'landmarks', 'crop_matrix'])
    ]

    ds_train_aug = ds_train.augment(aug_512)
    ds_test_common_aug = ds_test_common.augment(aug_512)
    ds_test_challenging_aug = ds_test_challenging.augment(aug_512)

    freeze(ds_test_common_aug, os.path.join(data_root, 'IBUG300W.test_common.zip'),
           {'image': storage.IMAGE_JPG}, with_prog=True)
    freeze(ds_test_challenging_aug, os.path.join(data_root, 'IBUG300W.test_challenging.zip'),
           {'image': storage.IMAGE_JPG}, with_prog=True)
    freeze(ds_train_aug, os.path.join(data_root, 'IBUG300W.train.zip'),
           {'image': storage.IMAGE_JPG}, with_prog=True)


def freeze_wflw(data_root):
    print('Freezing WFLW ...')
    ds_train = WFLW(os.path.join(data_root, 'WFLW'), split=TRAIN)
    print(f'train: {len(ds_train)}')
    ds_tests = dict()
    for subset in ['all', 'blur', 'expression', 'illumination', 'largepose', 'makeup', 'occlusion']:
        ds_tests[subset] = WFLW(os.path.join(
            data_root, 'WFLW'), split=TEST, subset=subset)
        print(f'test_{subset}: {len(ds_tests[subset])}')

    aug_512 = [
        aug.With(('box', None), 'crop_matrix', aug.UpdateCropAndResizeMatrix(
            (512, 512), align_corners=False)),
        aug.Filter(['image', 'label', 'landmarks', 'crop_matrix'])
    ]

    ds_train_aug = ds_train.augment(aug_512)
    ds_tests_aug = {subset: ds_test.augment(
        aug_512) for subset, ds_test in ds_tests.items()}

    freeze(ds_train_aug, os.path.join(data_root, 'WFLW.train.zip'),
           {'image': storage.IMAGE_JPG}, with_prog=True)
    for subset, ds_test_aug in ds_tests_aug.items():
        freeze(ds_test_aug, os.path.join(data_root, f'WFLW.test_{subset}.zip'),
               {'image': storage.IMAGE_JPG}, with_prog=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str)
    parser.add_argument('--dataset', default='all', type=str)

    args = parser.parse_args()

    if args.dataset in {'all', 'lapa'}:
        freeze_lapa(args.data_root)
    if args.dataset in {'all', 'celebamaskhq'}:
        freeze_celebamaskhq(args.data_root)
    if args.dataset in {'all', 'aflw19'}:
        freeze_aflw19(args.data_root)
    if args.dataset in {'all', 'ibug300w'}:
        freeze_ibug300w(args.data_root)
    if args.dataset in {'all', 'wflw'}:
        freeze_wflw(args.data_root)
