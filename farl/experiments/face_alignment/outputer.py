from typing import Mapping
import numpy as np
import cv2
import os

from blueprint.ml.viz import draw_landmarks


def _denormalize_points(normalized_points: np.ndarray, h: int, w: int) -> np.ndarray:
    """ Reverse normalize_points.

    Args:
        normalized_points: array of shape npoints x 2.
    """
    return normalized_points * np.array([[w, h]], dtype=normalized_points.dtype) - 0.5


class FaceAlignmentOutputer:
    def __init__(self, outputs_dir: str, image_tag: str = 'image',
                 pred_landmark_tag: str = 'pred_landmark',
                 draw_point_radius: int = 5) -> None:
        self.outputs_dir = outputs_dir
        self.image_tag = image_tag
        self.pred_landmark_tag = pred_landmark_tag
        self.draw_point_radius = draw_point_radius

    def __call__(self, eval_dataset_name: str, data: Mapping[str, np.ndarray]):
        # print(data.keys())
        assert 'sample_name' in data
        sample_name = data['sample_name']
        image = data[self.image_tag]  # 3 x h x w
        image = np.array((image * 255).astype(np.uint8))  # h x w x 3
        pred_landmark = data[self.pred_landmark_tag]  # npoints x 2

        # h, w, _ = image.shape
        # print(image.shape, image.dtype, image.max(), image.min())
        # pred_landmark = _denormalize_points(pred_landmark, h, w)
        # print(pred_landmark.shape, pred_landmark.max(), pred_landmark.min())
        image = draw_landmarks(image, pred_landmark, color=[
                               255]*4, radius=self.draw_point_radius)

        out_dir = os.path.join(self.outputs_dir, eval_dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f'{sample_name}.png'), cv2.cvtColor(
            image, cv2.COLOR_RGB2BGR))
