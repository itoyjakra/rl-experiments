import numpy as np
import torch


def process_pong_frame(image):
    """Returns a 80x80 pixel a single pong frame.
    Input image is 160x160x3
    """
    return np.mean(image[34:-16:2, ::2] - np.array([244, 72, 17]), -1) / 255
    return torch.from_numpy(frame)
