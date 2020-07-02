import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import LOSSES


@LOSSES.register_module
class ImageGradientLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(ImageGradientLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, gray_image, pred):
        """
        :param gray_image: gray gt_bboxes normalized to [0, 1] with shape (n,h,w)
        :param pred: predicted masks with size (n,h,w)
        :return:
        """
        gray_image = gray_image.unsqueeze(1).float()
        pred = pred.unsqueeze(1).float()
        gradient_tensor_x = torch.Tensor([[1.0, 0.0, -1.0],
                                          [2.0, 0.0, -2.0],
                                          [1.0, 0.0, -1.0]]).to(pred.device).view((1, 1, 3, 3))

        gradient_tensor_y = torch.Tensor([[1.0, 2.0, 1.0],
                                          [0.0, 0.0, 0.0],
                                          [-1.0, -2.0, -1.0]]).to(pred.device).view((1, 1, 3, 3))

        I_x = F.conv2d(gray_image, gradient_tensor_x, padding=1)
        G_x = F.conv2d(pred, gradient_tensor_x, padding=1)

        I_y = F.conv2d(gray_image, gradient_tensor_y, padding=1)
        G_y = F.conv2d(pred, gradient_tensor_y, padding=1)

        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2) + 1e-6)
        gradient = 1 - torch.pow(I_x * G_x + I_y * G_y, 2)

        image_gradient_loss = torch.sum(torch.mul(G, gradient)) / torch.sum(G)
        image_gradient_loss = torch.clamp_min(image_gradient_loss, 0)
        if self.reduction == "mean":
            image_gradient_loss = image_gradient_loss.mean()
        elif self.reduction == "sum":
            image_gradient_loss = image_gradient_loss.sum()

        return image_gradient_loss * self.loss_weight


@LOSSES.register_module
class BoundaryGradientLoss(nn.Module):
    def __init__(self,
                 lamb=1.5,
                 alpha=0.5,
                 reduction='mean',
                 loss_weight=1.0):
        super(BoundaryGradientLoss, self).__init__()
        self.lamb = lamb
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, gray_image, pred, mask):
        """
        :param gray_image: gray gt_bboxes normalized to [0, 1] with shape (n,h,w)
        :param pred: predicted masks with size (n,h,w)
        :param mask: boundary mask (can be generate from ground truth foreground mask by morphological transformation)
        :return:
        """
        gray_image = gray_image.unsqueeze(1).float()
        pred = pred.unsqueeze(1).float()

        gradient_tensor_x = torch.Tensor([[1.0, 0.0, -1.0],
                                          [2.0, 0.0, -2.0],
                                          [1.0, 0.0, -1.0]]).to(pred.device).view((1, 1, 3, 3))

        gradient_tensor_y = torch.Tensor([[1.0, 2.0, 1.0],
                                          [0.0, 0.0, 0.0],
                                          [-1.0, -2.0, -1.0]]).to(pred.device).view((1, 1, 3, 3))

        I_x = F.conv2d(gray_image, gradient_tensor_x, padding=1)
        G_x = F.conv2d(pred, gradient_tensor_x, padding=1)

        I_y = F.conv2d(gray_image, gradient_tensor_y, padding=1)
        G_y = F.conv2d(pred, gradient_tensor_y, padding=1)

        I = torch.sqrt(torch.pow(I_x, 2) + torch.pow(I_y, 2) + 1e-6)
        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2) + 1e-6)

        I_v = (I_x / I, I_y / I)
        G_v = (G_x / G, G_y / G)

        Lcos = (1 - torch.abs(I_v[0] * G_v[0] + I_v[1] * G_v[1])) * G
        Lmag = torch.clamp_min(self.lamb * I - G, 0)

        boundary_gradient_loss = (self.alpha * Lcos + (1 - self.alpha) * Lmag) * mask
        if self.reduction == "mean":
            boundary_gradient_loss = boundary_gradient_loss.mean()
        elif self.reduction == "sum":
            boundary_gradient_loss = boundary_gradient_loss.sum()

        return boundary_gradient_loss * self.loss_weight
