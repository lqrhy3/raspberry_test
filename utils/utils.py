import torch
import numpy as np


class ImageToTensor:
    def __call__(self, img):
        # opencv image: W x H x C
        # torch image: C x W x H
        image = img.transpose((2, 0, 1))

        return torch.from_numpy(image)


def get_most_confident_bbox(output, num_bbox):
    shape = output.shape
    flatten_idx = torch.argmax(output[:, [i * 5 + 4 for i in range(num_bbox)],  :, :])
    z = 4 + (5 * (flatten_idx // (shape[2] * shape[3])))
    xy = flatten_idx % (shape[2] * shape[3])
    x = xy // shape[3]
    y = xy % shape[3]

    return x, y, z - 4


def transform_bbox_coords(output, x, y, z, image_size, grid_size):
    coords = output[:, z:z+4, x, y].squeeze(0)
    cell_size = image_size[0] / grid_size
    coords[0] = cell_size * x + cell_size * coords[0]
    coords[1] = cell_size * y + cell_size * coords[1]
    coords[2] = image_size[0] * coords[2]
    coords[3] = image_size[1] * coords[3]
    return coords


def xywh2xyxy(coords):
    """Args:
           coords: np.array([x_center, y_center, w, h])
    """
    new_coords = np.empty((4,), dtype=np.int)
    new_coords[0] = coords[0] - coords[2] // 2
    new_coords[1] = coords[1] - coords[3] // 2
    new_coords[2] = coords[0] + coords[2] // 2
    new_coords[3] = coords[1] + coords[3] // 2
    return new_coords
