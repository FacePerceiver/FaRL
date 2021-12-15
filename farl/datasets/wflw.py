import os
import numpy as np
import cv2

from blueprint.ml import Dataset, Split


class WFLW(Dataset):
    def __init__(self, root, split=Split.ALL, subset='all'):
        self.root = root

        anno_file = None
        if split == Split.TRAIN:
            anno_file = 'face_landmarks_wflw_train.csv'
        elif split == Split.TEST:
            if subset == 'all':
                anno_file = 'face_landmarks_wflw_test.csv'
            else:
                anno_file = f'face_landmarks_wflw_test_{subset}.csv'

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

                landmarks = np.reshape(
                    np.array([float(v) for v in landmarks], dtype=np.float32), [98, 2])
                cx, cy = np.mean(landmarks, axis=0)

                sample_name = os.path.splitext(im_path)[0].replace(
                    '/', '.') + ('_%.3f_%.3f' % (cx, cy))
                im_path = os.path.join(self.root, 'WFLW_images', im_path)

                assert os.path.exists(im_path)                

                self.info_list.append({
                    'sample_name': sample_name,
                    'im_path': im_path,
                    'landmarks': landmarks,
                    'box_info': (float(scale), float(center_w), float(center_h))
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
