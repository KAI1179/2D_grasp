import torch
from skimage.filters import gaussian


def post_process_output(q_img, cos_img, sin_img, width_img, mask_img, center_img):
    """
    Post-process the raw output of the network, convert to numpy arrays, apply filtering.
    :param q_img: Q output of network (as torch Tensors)
    :param cos_img: cos output of network
    :param sin_img: sin output of network
    :param width_img: Width output of network
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0 #为什么要乘以150？
    mask_img = mask_img.cpu().numpy().squeeze()
    center_img = center_img.cpu().numpy().squeeze()

    mask_img[mask_img >= 0.5] = 1 #将mask区域全部转换为1
    mask_img[mask_img < 0.5] = 0

    center_img = gaussian(center_img, 1.0, preserve_range=True)
    q_img = q_img * center_img
    q_img = gaussian(q_img, 2.0, preserve_range=True)
    #ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    #width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img, mask_img
