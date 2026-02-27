import torch
import torch.nn as nn
import torch.nn.functional as F


def bce2d(input, target):
    n, c, h, w = input.size()
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()
    pos_index = (target_t == 1)
    neg_index = (target_t == 0)
    ignore_index = (target_t > 1)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0

    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    ignore_index = ignore_index.data.cpu().numpy().astype(bool)

    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num
    weight[ignore_index] = 0
    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss

class GraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width, y_mask, y_center = yc
        pos_pred, cos_pred, sin_pred, width_pred, or_pred, center_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)
        region_loss = bce2d(or_pred, y_mask)
        center_loss =  F.smooth_l1_loss(center_pred, y_center)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss + region_loss + center_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss,
                'region_loss':region_loss,
                'center_loss':center_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred,
                'object_region':or_pred,
                'center_pred':center_pred
            }
        }

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred, or_pred, center_pred = self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred,
            'object_region': or_pred,
            'center_pred': center_pred
        }


class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in
