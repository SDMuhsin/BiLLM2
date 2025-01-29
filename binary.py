from numpy import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


index = 0
@torch.no_grad()
def part_mean(tensor, op='-'):
    non_zero = tensor*(tensor!=0)

    mean_val = non_zero.mean(-1).view(-1, 1)

    return mean_val

@torch.no_grad()
def high_order_residual(x, mask, order=2):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        binary= torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask
    
    return sum_order

@torch.no_grad()
def normal_quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

@torch.no_grad()
def robust_high_order_residual(x, mask, order=2, clamp_factor=2.5):
    """
    A robust variant of residual binarization that clamps outliers at each iteration.
    x: the weight matrix (oc x ic)
    mask: boolean tensor indicating where to apply binarization
    order: number of residual binarization steps
    clamp_factor: multiple of std-dev for clamping outliers
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    for _ in range(order):
        # Compute the residual for this iteration
        residual = new_matrix - sum_order

        # Only consider masked elements
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'), device=residual.device))

        # Compute row-wise mean (ignoring NaNs)
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)

        # Center the residual around the mean
        centered_x = masked_x_tensor - mean_tensor_all[:, None]

        # Handle NaNs explicitly for standard deviation
        # Count valid (non-NaN) elements
        valid_counts = torch.sum(~torch.isnan(centered_x), dim=1).float()
        valid_counts = torch.where(valid_counts > 0, valid_counts, torch.tensor(1.0, device=residual.device))  # Avoid divide-by-zero

        # Compute squared deviations
        squared_deviations = torch.where(
            ~torch.isnan(centered_x), centered_x**2, torch.zeros_like(centered_x)
        )
        variance = torch.sum(squared_deviations, dim=1) / valid_counts
        std_tensor_all = torch.sqrt(variance)

        # Clamp outliers: anything beyond ±(clamp_factor * std)
        clamped_x_tensor = torch.clamp(
            centered_x,
            min=-clamp_factor * std_tensor_all[:, None],
            max=clamp_factor * std_tensor_all[:, None],
        )

        # Compute scale as mean(abs(.)) after clamping
        scale_tensor_all = torch.nanmean(torch.abs(clamped_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        # Binarize + scale + shift
        binary = torch.sign(clamped_x_tensor) * scale_tensor_all[:, None]
        binary = binary + mean_tensor_all[:, None]

        # Accumulate into sum_order
        sum_order += binary * mask

    return sum_order


@torch.no_grad()
def mest_robust_residual_binarization(x, mask, order=2, kappa=1.0):
    """
    Robust residual binarization with M-estimation style outlier handling.

    Parameters:
    - x (torch.Tensor): The weight matrix (rows = out_channels, cols = in_channels).
    - mask (torch.Tensor): Boolean mask indicating which entries to binarize.
    - order (int): Number of residual expansions.
    - kappa (float): Robustness parameter controlling outlier down-weighting.

    Returns:
    - torch.Tensor: The binarized approximation of `x`.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask

    for od in range(order):
        # Residual at the current iteration
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'), device=x.device))

        # Compute robust weights: f(r) = 1 / (1 + (r / kappa)^2)
        weight_all = 1.0 / (1.0 + (masked_x_tensor / kappa) ** 2)
        weight_all = torch.where(torch.isnan(weight_all), torch.zeros_like(weight_all), weight_all)

        # Weighted mean computation
        valid_tensor = torch.logical_not(torch.isnan(masked_x_tensor))
        wsum = torch.nansum(weight_all * valid_tensor, dim=1, keepdim=True) + 1e-8
        weighted_vals = torch.where(valid_tensor, masked_x_tensor * weight_all, torch.zeros_like(masked_x_tensor))
        mean_tensor_all = torch.nansum(weighted_vals, dim=1, keepdim=True) / wsum

        # Subtract the robust mean
        masked_x_tensor = masked_x_tensor - mean_tensor_all

        # Compute robust scale
        abs_diff = torch.abs(masked_x_tensor) * weight_all
        scale_tensor_all = torch.nansum(abs_diff, dim=1, keepdim=True) / wsum

        # Replace NaNs with zeros
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        # Binarization step: sign(r') * scale + mean
        binary = torch.sign(masked_x_tensor)
        binary = binary * scale_tensor_all
        binary = binary + mean_tensor_all

        # Accumulate the result
        sum_order = sum_order + torch.where(mask, binary, torch.zeros_like(binary))

    return sum_order

@torch.no_grad()
def median_high_order_residual(x, mask, order=2):
    """
    Proposed robust residual binarization (medianbraq).
    Uses median-based offset and scale (median absolute deviation).
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # Median-based offset
        # nanmedian returns (values, indices), so we take the .values.
        median_tensor_all = torch.nanmedian(masked_x_tensor, dim=1).values
        median_tensor_all = torch.where(torch.isnan(median_tensor_all),
                                        torch.zeros_like(median_tensor_all),
                                        median_tensor_all)

        # Subtract offset
        masked_x_tensor -= median_tensor_all[:, None]

        # Scale = median of absolute values (robust to outliers)
        abs_masked = torch.abs(masked_x_tensor)
        scale_tensor_all = torch.nanmedian(abs_masked, dim=1).values
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all),
                                       torch.zeros_like(scale_tensor_all),
                                       scale_tensor_all)

        # Binarize
        binary = torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += median_tensor_all[:, None]
        sum_order = sum_order + binary*mask
    
    return sum_order


class Binarization(nn.Module):
    def __init__(self, weight, method="2bit", groupsize=-1):
        super().__init__()
        oc,ic=weight.shape
        if groupsize==-1:
            groupsize=ic
        self.groupsize=groupsize
        self.n_groups=math.ceil(ic/groupsize)
        self.method=method
        self.mean = 0
        # Add defaults for the (2) mest robust method
        self.kappa = 1.0  # Robustness parameter
        self.order = 2    # Number of residual expansions
    def quantize(self, w, mask, order=2, groupi=0):
        if self.method=="xnor":
            w_mean = self.mean[groupi]
            w = w - w_mean  # oc, ic
            w = w.sign()
            w = w * self.scale[groupi]
            w+=w_mean
        elif self.method=="braq": # The method used in paper
            w = high_order_residual(w, mask, order=order)
        elif self.method=="robq":  # Our robust varianti (1)
            w = robust_high_order_residual(w, mask, order=order, clamp_factor=2.5)
        elif self.method == "mestrobq":  # New robust method
            w = mest_robust_residual_binarization(w, mask, order=self.order, kappa=self.kappa)
        elif self.method == "medianbraq":  # New robust method
            w = median_high_order_residual(w, mask, order=self.order)
        elif self.method=="sign":
            w=(w>0).float()
            w*=self.scale[groupi]
        elif self.method=="rtn":
            w=F.relu(w)
            w_int=(w/self.scale[groupi]).round().clamp(0,1)
            w=w_int*self.scale[groupi]
        elif self.method in ['2bit','4bit']:

            bits = int(self.method[0])
            perchannel = True
            weight = True
            dev = w.device
            maxq = torch.tensor(2 ** bits - 1)
            scale = torch.zeros(1)
            zero = torch.zeros(1)

            if dev != scale.device:
                scale=scale.to(dev)
                zero=zero.to(dev)
                maxq=maxq.to(dev)

            x = w.clone()
            shape = x.shape

            if perchannel:
                if weight:
                    x = x.flatten(1)
                else:
                    if len(shape) == 4:
                        x = x.permute([1, 0, 2, 3])
                        x = x.flatten(1)
                    if len(shape) == 3:
                        x = x.reshape((-1, shape[-1])).t()
                    if len(shape) == 2:
                        x = x.t()
            else:
                x = x.flatten().unsqueeze(0)
            tmp = torch.zeros(x.shape[0], device=dev)
            xmin = torch.minimum(x.min(1)[0], tmp)
            xmax = torch.maximum(x.max(1)[0], tmp)

            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)
            if not perchannel:
                if weight:
                    tmp = shape[0]
                else:
                    tmp = shape[1] if len(shape) != 3 else shape[2]
                scale = scale.repeat(tmp)
                zero = zero.repeat(tmp)

            if weight:
                shape = [-1] + [1] * (len(shape) - 1)
                scale = scale.reshape(shape)
                zero = zero.reshape(shape)
            w = normal_quantize(w, scale, zero, maxq)

        elif self.method=="prune":
            return torch.zeros_like(w)
        return w
