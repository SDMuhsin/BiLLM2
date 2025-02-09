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
    new_matrix = x.clone() # Keep a copy of the original weight matrix x
    new_matrix = new_matrix * mask # Pick out only salient columsn
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'))) # Use only valid positions of residual (invalids are marked with nan)

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1) #  Row wise mean of residual
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        masked_x_tensor -= mean_tensor_all[:, None] # Subtracts mean from each row (only valid rows) : Centers all elements at 0
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1) # Gets averagei absolute value, row wise = estimate of alpha
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        binary= torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None] # Rescale (alpha * B)
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask # Add X = ... + Bk * alpha_k
    
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

@torch.no_grad()
def orthogonal_residual(x, mask, order=2):
    """
    Orthogonal Residual Binarization (ORB)

    This patched version handles fully masked-out rows
    by replicating 'high_order_residual' logic—any row
    with no unmasked elements becomes zero instead of NaN.
    """

    sum_order = torch.zeros_like(x)
    expansions = []
    
    for od in range(order):
        # Residual to approximate
        residual = x - sum_order

        # Mark unmasked elements; others = NaN
        masked_residual = torch.where(mask, residual, torch.tensor(float('nan'), device=x.device))
        
        # Row-wise mean (ignoring NaNs)
        mean_val = torch.nanmean(masked_residual, dim=1, keepdim=True)
        # If the entire row is NaN => force that mean to 0
        mean_val = torch.where(torch.isnan(mean_val),
                               torch.zeros_like(mean_val),
                               mean_val)
        
        # Center the residual around the mean
        centered = masked_residual - mean_val
        
        # Convert all masked-out elements to 0 (not NaN)
        centered = torch.where(mask, centered, torch.zeros_like(centered))

        # Orthogonal projection against previous expansions
        for exp in expansions:
            dot_num = (centered * exp).mean(dim=1, keepdim=True)
            dot_den = (exp * exp).mean(dim=1, keepdim=True) + 1e-12
            proj = dot_num * exp / dot_den
            centered = centered - proj
        
        # Row-wise scaling
        scale_val = torch.nanmean(torch.abs(centered), dim=1, keepdim=True)
        # If row is all zero => NaN => set to 0
        scale_val = torch.where(torch.isnan(scale_val),
                                torch.zeros_like(scale_val),
                                scale_val)
        
        # Sign + scale + shift
        binary = torch.sign(centered) * scale_val + mean_val
        
        # Update expansions & sum
        expansions.append(binary)
        sum_order = sum_order + binary * mask

    return sum_order

@torch.no_grad()
def weighted_high_order_residual(x, mask, order=2):
    """
    Weighted Residual Binarization (WHOR):
    Iteratively approximates 'x' with a sum of binary expansions, 
    weighting errors by their magnitude so that large residuals 
    get reduced more aggressively.
    
    The final bit cost is the same as standard residual binarization:
    we do 'order' expansions, each storing one mean + one scale + sign-bits.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask  # only operate within the valid region

    for od in range(order):
        # 1) Compute residual
        residual = new_matrix - sum_order

        # 2) Mask out invalid positions
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'), device=x.device))

        # 3) Define weights = abs(residual). 
        #    This emphasizes elements with large residuals.
        w = torch.abs(masked_x_tensor)

        # 4) Weighted mean calculation: 
        #    mean = (sum_i w_i * r_i) / (sum_i w_i)
        numerator = torch.nansum(w * masked_x_tensor, dim=1, keepdim=True)
        denominator = torch.nansum(w, dim=1, keepdim=True) + 1e-8
        mean_tensor_all = numerator / denominator

        # 5) Subtract the mean from residual
        masked_x_tensor = masked_x_tensor - mean_tensor_all

        # 6) Weighted scale:
        #    scale = (sum_i w_i * |r_i - mean|) / (sum_i w_i)
        scale_numerator = torch.nansum(w * torch.abs(masked_x_tensor), dim=1, keepdim=True)
        scale_tensor_all = scale_numerator / denominator

        # 7) Form the binary expansion: sign + scale + mean
        binary = torch.sign(masked_x_tensor) * scale_tensor_all
        binary = binary + mean_tensor_all

        # 8) Accumulate into sum_order
        sum_order = sum_order + binary * mask

    return sum_order

@torch.no_grad()
def attenuated_residual(x, mask, order=2, gamma=0.5):
    """
    Attenuated Residual Binarization (ARB)
    - Similar to `high_order_residual` (braq)
    - Each iteration's binary correction is damped by a factor gamma.
    - Retains 1-bit expansions and same memory overhead as braq.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask

    for od in range(order):
        residual = new_matrix - sum_order
        # Mask out the elements
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'), device=x.device))

        # Compute row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)

        # Center
        masked_x_tensor -= mean_tensor_all[:, None]

        # Compute row-wise average absolute deviation for scale
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        # Sign + scale + shift
        binary = torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]

        # Instead of subtracting full `binary`, we only subtract gamma * binary
        # and accumulate the partial correction
        sum_order = sum_order + gamma * binary * mask

    return sum_order
@torch.no_grad()
def balanced_high_order_residual(x, mask, order=2):
    """
    Balanced Residual Binarization in multiple passes.
    Enforces ~0 net sum in each pass by balancing +1/−1.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask  # only keep valid weights

    for od in range(order):
        residual = new_matrix - sum_order
        # masked view: ignore positions where mask=0
        masked_residual = torch.where(mask, residual, torch.tensor(float('nan'), device=x.device))

        # Compute row-wise scale (same as braq's L2-optimal alpha)
        alpha = torch.nanmean(torch.abs(masked_residual), dim=1)
        alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)  # safe fallback

        # Balanced sign step per row
        B = torch.zeros_like(masked_residual)

        for i in range(B.shape[0]):
            row = masked_residual[i]
            valid_mask = ~torch.isnan(row)
            valid_vals = row[valid_mask]

            if valid_vals.numel() == 0:
                continue

            sorted_vals, sorted_idx = torch.sort(valid_vals)
            n_valid = len(sorted_vals)
            half = n_valid // 2  # integer division

            # Ensure `torch.arange()` is created on the same device as `row`
            row_indices = torch.arange(len(row), device=row.device)[valid_mask][sorted_idx]

            B[i, row_indices[:half]] = -1
            B[i, row_indices[half:]] = 1

        # Convert B from masked_residual shape to a real tensor (NaN->0 for masked positions)
        B = torch.where(mask, B, torch.zeros_like(B))

        # Weighted update = alpha_i * B
        alpha = alpha.view(-1, 1)
        sum_order = sum_order + alpha * B

    return sum_order
@torch.no_grad()
def joint_residual_binarization(x, mask, iters=3):
    """
    Jointly refines two binary expansions (B1, B2) with scales (alpha1, alpha2)
    by coordinate-descent style updates, seeking to minimize || x - alpha1 B1 - alpha2 B2 ||^2
    without adding new storage. Returns final sum_order = alpha1*B1 + alpha2*B2.
    """
    # 1) Initialize with standard first-pass binarization
    x_local = x.clone() * mask
    # B1, alpha1
    mean1 = torch.nanmean(torch.where(mask, x_local, torch.tensor(float('nan'))), dim=1)
    mean1 = torch.where(torch.isnan(mean1), torch.zeros_like(mean1), mean1)
    x_shifted = x_local - mean1[:, None]
    alpha1 = torch.nanmean(torch.abs(x_shifted), dim=1)
    alpha1 = torch.where(torch.isnan(alpha1), torch.zeros_like(alpha1), alpha1)
    B1 = torch.sign(x_shifted)

    # 2) Initialize B2, alpha2 from the residual
    R = x_local - (B1 * alpha1[:, None])
    mean2 = torch.nanmean(torch.where(mask, R, torch.tensor(float('nan'))), dim=1)
    mean2 = torch.where(torch.isnan(mean2), torch.zeros_like(mean2), mean2)
    R_shifted = R - mean2[:, None]
    alpha2 = torch.nanmean(torch.abs(R_shifted), dim=1)
    alpha2 = torch.where(torch.isnan(alpha2), torch.zeros_like(alpha2), alpha2)
    B2 = torch.sign(R_shifted)

    # 3) Iterative refinement
    #    Re-fit B1 once B2 is known, then re-fit B2, etc.
    for _ in range(iters):
        # Recompute residual ignoring B2
        R1 = x_local - (B2 * alpha2[:, None])
        # Fit B1, alpha1 again
        mean1 = torch.nanmean(torch.where(mask, R1, torch.tensor(float('nan'))), dim=1)
        mean1 = torch.where(torch.isnan(mean1), torch.zeros_like(mean1), mean1)
        R1_shifted = R1 - mean1[:, None]
        alpha1 = torch.nanmean(torch.abs(R1_shifted), dim=1)
        alpha1 = torch.where(torch.isnan(alpha1), torch.zeros_like(alpha1), alpha1)
        B1 = torch.sign(R1_shifted)

        # Now re-fit B2 from the new B1
        R2 = x_local - (B1 * alpha1[:, None])
        mean2 = torch.nanmean(torch.where(mask, R2, torch.tensor(float('nan'))), dim=1)
        mean2 = torch.where(torch.isnan(mean2), torch.zeros_like(mean2), mean2)
        R2_shifted = R2 - mean2[:, None]
        alpha2 = torch.nanmean(torch.abs(R2_shifted), dim=1)
        alpha2 = torch.where(torch.isnan(alpha2), torch.zeros_like(alpha2), alpha2)
        B2 = torch.sign(R2_shifted)

    # Final combination
    sum_order = (B1 * alpha1[:, None] + mean1[:, None]) \
              + (B2 * alpha2[:, None] + mean2[:, None])
    sum_order = sum_order * mask
    return sum_order

@torch.no_grad()
def coupled_residual_binarization(x, mask, order=2):
    """
    Performs a two-binary-expansion approximation (like braq) but
    co-optimizes the scale factors alpha_1, alpha_2 in closed form.

    x:     (oc, ic) weight matrix
    mask:  boolean mask with same shape as x
    order: number of expansions (we only implement 2 expansions here)

    Returns: sum_order -> final approximate binarized matrix
    """
    # We will do this row by row. 
    # For each row, we:
    #   1) Subtract mean.
    #   2) Get B1, alpha_1 from sign/average of magnitude.
    #   3) Form residual, get B2, alpha_2 similarly.
    #   4) Solve for alpha_1, alpha_2 simultaneously in closed form.
    #   5) Re-add the mean.
    
    # Make a clone of x that we will modify
    new_matrix = x.clone()
    new_matrix = new_matrix * mask  # only consider valid entries

    # We'll accumulate the final approximation in sum_order
    sum_order = torch.zeros_like(new_matrix)

    oc, ic = new_matrix.shape

    # Row-wise processing
    for row_idx in range(oc):
        # Extract row and mask
        row = new_matrix[row_idx, :]
        row_mask = mask[row_idx, :]
        
        # If nothing is masked-in, skip
        if not torch.any(row_mask):
            continue
        
        # Grab just the valid elements for the masked row
        row_vals = row[row_mask]
        
        # 1) Subtract mean
        row_mean = row_vals.mean()
        centered = row_vals - row_mean
        
        # 2) First pass: B1, alpha_1
        B1 = torch.sign(centered)
        alpha_1 = centered.abs().mean()
        
        # 3) Residual, second pass B2, alpha_2
        r = centered - alpha_1 * B1
        B2 = torch.sign(r)
        alpha_2 = r.abs().mean()
        
        if order >= 2:
            # 4) Solve for alpha_1, alpha_2 in closed form
            #    We define:
            #       d = # valid elements
            #       c12 = sum(B1 * B2)
            #       c1w = sum(centered * B1)
            #       c2w = sum(centered * B2)
            d = float(row_vals.numel())
            c12 = torch.sum(B1 * B2).item()
            c1w = torch.sum(centered * B1).item()
            c2w = torch.sum(centered * B2).item()
            
            det = d*d - c12*c12
            if abs(det) > 1e-12:
                alpha_1_new = ( c1w*d - c2w*c12 ) / det
                alpha_2_new = ( c2w*d - c1w*c12 ) / det
                # alpha should be non-negative, so we clamp to >= 0
                alpha_1 = max(alpha_1_new, 0.0)
                alpha_2 = max(alpha_2_new, 0.0)

        # 5) Reconstruct final row approximation
        approx = row_mean + alpha_1 * B1 + alpha_2 * B2
        
        # Place this approximation back into sum_order at masked positions
        out_row = sum_order[row_idx, :]
        out_row[row_mask] = approx

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
        elif self.method=="jrb":  # <-- NEW PROPOSAL
            w = joint_residual_binarization(w, mask, iters=order)
        elif self.method=="crb":  # <-- NEW PROPOSAL
            w = coupled_residual_binarization(w, mask, order=order)
        elif self.method=="bhor": # T
            w = balanced_high_order_residual(w, mask, order=order)
        elif self.method=="orb": # Orthogonal Residual Binarization
            w = orthogonal_residual(w, mask, order=order)
        elif self.method=="arb":
            w = attenuated_residual(w, mask, order=order, gamma=0.8)
        elif self.method=="whor": # Weighted High Order Residual
            w = weighted_high_order_residual(w, mask, order=order)
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
