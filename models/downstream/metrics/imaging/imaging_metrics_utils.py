from typing import Union, Any

import torch
import numpy as np
from scipy.ndimage import label
from scipy.spatial.distance import pdist
from torch import Tensor



def compute_confusion_matrix(predicted: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute a confusion matrix for predicted and ground truth tensors.

    Parameters:
        predicted (torch.Tensor): Predicted class indices
        gt (torch.Tensor): Ground truth class indices
        num_classes (int): Number of classes

    Returns:
        torch.Tensor: Confusion matrix of shape [num_classes, num_classes]
                      Rows = true classes, Columns = predicted classes
    """
    # Flatten the tensors
    predicted_flat = predicted.view(-1)  # makes the tensor a row of numbers
    gt_flat = gt.view(-1)

    # Initialize confusion matrix
    confusion_matrix = torch.zeros((1, num_classes, num_classes), dtype=torch.float32)
    # Fill confusion matrix
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            mask = (gt_flat == true_class) & (predicted_flat == pred_class)
            confusion_matrix[0, true_class, pred_class] = torch.sum(mask).float() / gt.shape[0]

    return confusion_matrix


def compute_loc_error(predicted: torch.Tensor, gt: torch.Tensor, predicted_lumps: torch.Tensor,
                      true_lumps: torch.Tensor, inference_on_big_phantom: bool = False, lump_value: int = 2, ) -> torch.Tensor:
    """
    Compute the location error between predicted and ground truth tensors.

    Parameters
    ----------
    predicted (torch.Tensor): Predicted class indices
    gt (torch.Tensor): Ground truth class indices
    predicted_lumps (torch.Tensor): the amount of lumps present in the predicted tensor, shape [B].
    true_lumps (torch.Tensor): the amount of lumps present in the ground truth tensor, shape [B].
    lump_value (int): the class index of the lump

    Returns
    -------
    loc_error (torch.Tensor): A tensor which holds the loc_error between predicted and ground truth tensors.

    """

    B = gt.shape[0]
    loc_error = 0
    for i in range(B):
        if true_lumps[i] == 0 and predicted_lumps[i] == 0:
            continue  # perfect guess

        if true_lumps[i] == 0 or predicted_lumps[i] == 0:
            loc_error += 10  # penalty
            continue

        pred_center = torch.argwhere(predicted[i] == lump_value).float().mean(dim=0)
        real_center = torch.argwhere(gt[i] == lump_value).float().mean(dim=0)

        H, W = gt.shape[-2:]

        three_dim = predicted[i].ndim == 3

        pred_center = mri_pixel_to_mm(pred_center, W, H, three_dim=three_dim, inference_on_big_phantom=inference_on_big_phantom)
        real_center = mri_pixel_to_mm(real_center, W, H, three_dim=three_dim, inference_on_big_phantom=inference_on_big_phantom)

        loc_error += torch.linalg.norm(pred_center - real_center).item()

    loc_error = torch.tensor(loc_error, dtype=torch.float32)

    return loc_error


def mri_pixel_to_mm(com: torch.Tensor, new_size_x: int, new_size_y: int, three_dim: bool, inference_on_big_phantom: bool) -> torch.Tensor:
    """
    Correct the coordinates for voxel spacing differences
    in MRI scans so that distances are expressed in millimeters.

    Parameters
    ----------
    com : (torch.Tensor): holds the point in the (z, y, x) format or (y, x) format.
    new_size_x : (int)
        The size of the volume along the x-axis (width) in voxels.
    new_size_y : (int)
        The size of the volume along the y-axis (height) in voxels.
    three_dim : (bool)
        if true 3 dimensions else 2.

    Returns
    -------
    new_com (torch.Tensor):
        Corrected coordinates in millimeters (z_mm, y_mm, x_mm) or (y_mm, x_mm).
    """

    original_size = 80

    # the numbers are the distance between pixel in the mri scans
    new_spacing_x = 0.763889 * (original_size / new_size_x)
    new_spacing_y = 0.763889 * (original_size / new_size_y)
    spacing_z = 0.9

    if inference_on_big_phantom:
        new_spacing_x *= 2
        new_spacing_y *= 2

    if three_dim:
        z, y, x = com
        new_com = (z * spacing_z, y * new_spacing_y, x * new_spacing_x)
    else:
        y, x = com
        new_com = (y * new_spacing_y, x * new_spacing_x)

    new_com = torch.tensor(new_com, dtype=torch.float32)

    return new_com


def get_diameter(tensor: torch.Tensor, inference_on_big_phantom, class_idx: int) -> torch.Tensor:
    """
    Finds the longest distance (diameter) in a tensor, per slic, per connected component.
    Parameters
    ----------
    tensor : (torch.Tensor)  torch.Tensor of shape [B, D, H, W]
    class_idx (int): The index of the class, 2 - pillar, 3 - lump.
        All other values are invalid.
    Returns
    -------
    torch.Tensor of shape [B] where each entry is the mean diameter in that img
    """
    IDX = [2, 3]
    if class_idx not in IDX:
        raise ValueError("type_idx must be 2 or 3.")

    tensor_np = tensor.cpu().numpy()
    if tensor_np.ndim == 3:
        tensor_np = tensor_np[:, None, :, :]  # add D=1
    elif tensor_np.ndim != 4:
        raise ValueError("tensor must be [B, H, W] or [B, D, H, W]")

    B, D, H, W = tensor_np.shape
    diameters = np.zeros((B, D))
    for b in range(B):
        for d in range(D):
            img = (tensor_np[b, d] == class_idx).astype(np.uint8)
            labeled, num = label(img)  # find connected components

            max_d = 0.0
            for i in range(1, num + 1):
                coords = np.argwhere(labeled == i)
                if len(coords) < 2:
                    continue  # skip single points

                # compute pairwise distances
                dists = pdist(coords, metric='euclidean')
                max_d = max(max_d, dists.max())

            diameters[b, d] = max_d

    diameters = diameters.mean(axis=1)

    # correct the coordinates
    original_size = 80
    scale = 0.763889 * (original_size / tensor_np.shape[2])

    if inference_on_big_phantom:
        scale *= 2

    diameters = diameters * scale

    return torch.tensor(diameters)

def diameter_error(predicted: torch.Tensor, gt: torch.Tensor, inference_on_big_phantom, class_idx: int) -> tuple[Tensor, Tensor]:
    """
    Compute the pillar/ lump error between predicted and ground truth 3D volumes.

    Parameters
    ----------
        predicted (torch.Tensor): Predicted 3D volume, shape [B, D, H, W].
        gt (torch.Tensor): Ground truth 3D volume, shape [B, D, H, W].
        class_idx (int): The index of the class, 2 - pillar, 3 - lump.
                All other values are invalid.

    Returns
    -------
        torch.Tensor: tensor representing the diameter rmse error and a tensor representing the
            diameter relative error.
    """
    distance_pred = get_diameter(predicted, inference_on_big_phantom, class_idx=class_idx)
    distance_gt = get_diameter(gt, inference_on_big_phantom, class_idx=class_idx)

    err = distance_gt - distance_pred
    rmse = torch.sqrt(err ** 2)

    rel_err = torch.abs(err) / (0.5 * (distance_gt + distance_pred))
    rel_err[torch.isnan(rel_err)] = 0.0
    return rmse, rel_err

