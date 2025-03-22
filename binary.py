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
def D_coupled_residual_binarization(x, mask, order=2):

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

index = 0

@torch.no_grad()
def coupled_residual_binarization(x, mask, order=2):
    """
    A unified binarization function that:
      - For order == 1: Performs a single-pass binarization (original simple approach).
      - For order >= 2: Performs a coupled two-expansion binarization (new approach),
        which jointly solves for the two scale factors.
    """
    global index
    index += 1

    # We'll always create sum_order and clone x
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask

    # ---------------------------
    # Case 1: order == 1
    # ---------------------------
    if order == 1:
        # Exactly the old single‐pass binarization
        residual = new_matrix - sum_order
        # Keep only valid positions
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan'), device=residual.device))

        # Row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Subtract row-wise mean
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale = average absolute value
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )

        # Binary = sign(masked_x_tensor)
        binary = torch.sign(masked_x_tensor)
        # Multiply by alpha
        binary *= scale_tensor_all[:, None]
        # Then add back the row mean
        binary += mean_tensor_all[:, None]

        # Add to sum_order for final approximation
        sum_order = sum_order + binary * mask

        return sum_order

    # ---------------------------
    # Case 2: order == 2
    # ---------------------------
    else:
        """
        Coupled two-expansion binarization:
          w ~ alpha1 * B1 + alpha2 * B2 + row_mean
        with alpha1, alpha2 solved jointly.
        """

        oc, ic = new_matrix.shape

        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                # If mask is all false in this row, skip
                continue

            row_vals = new_matrix[row_idx, row_mask]

            # 1) Subtract row mean
            row_mean = row_vals.mean()
            centered = row_vals - row_mean

            # 2) First expansion: B1, alpha1
            B1 = torch.sign(centered)
            alpha1 = centered.abs().mean()

            # 3) Second expansion: B2, alpha2 w.r.t. residual
            r = centered - alpha1 * B1
            B2 = torch.sign(r)
            alpha2 = r.abs().mean()

            # 4) Solve alpha1, alpha2 in closed form simultaneously
            #    Minimizing || centered - alpha1 B1 - alpha2 B2 ||^2
            d = float(row_vals.numel())
            c12 = torch.sum(B1 * B2).item()
            c1w = torch.sum(centered * B1).item()
            c2w = torch.sum(centered * B2).item()

            det = d * d - c12 * c12
            if abs(det) > 1e-12:
                # Solve the 2x2 linear system for alpha1, alpha2
                alpha1_new = ( c1w * d - c2w * c12 ) / det
                alpha2_new = ( c2w * d - c1w * c12 ) / det
                # Constrain to be non-negative
                alpha1 = max(alpha1_new, 0.0)
                alpha2 = max(alpha2_new, 0.0)

            # 5) Final approximation for that row
            approx_row = row_mean + alpha1 * B1 + alpha2 * B2
            
            # Put it back into sum_order
            sum_order[row_idx, row_mask] = approx_row

        return sum_order

@torch.no_grad()
def coupled_residual_binarization_stable(x, mask, order=2, lam=1e-5):
    """
    A unified binarization function with Tikhonov stabilization for the 2-expansion case.

    Args:
      x (tensor): weight matrix, shape (oc, ic)
      mask (tensor, bool): True where weights are to be binarized, False otherwise
      order (int): 
         - 1 => single-pass binarization: w ~ alpha * sign(w - mean)
         - >=2 => two-expansion binarization with Tikhonov-stabilized
                  closed-form for alpha1, alpha2
      lam (float): Tikhonov regularization strength (rho).
                   By default 1e-5 is used; 
                   you may tune this if alphas are still 0 or negative too frequently.

    Returns:
      sum_order (tensor): the binarized approximation of x
    """
    # We'll always create sum_order and clone x
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask

    # A small global index if needed
    # (optional, matching your existing pattern)
    global index
    index += 1

    # ---------------------------
    # Case 1: order == 1
    # ---------------------------
    if order == 1:
        # Exactly the old single-pass binarization
        residual = new_matrix - sum_order
        # Keep only valid positions
        masked_x_tensor = torch.where(
            mask, 
            residual, 
            torch.tensor(float('nan'), device=residual.device)
        )

        # Row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Subtract row-wise mean
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale = average absolute value
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )

        # Binary = sign(masked_x_tensor)
        binary = torch.sign(masked_x_tensor)
        # Multiply by alpha
        binary *= scale_tensor_all[:, None]
        # Then add back the row mean
        binary += mean_tensor_all[:, None]

        # Add to sum_order for final approximation
        sum_order = sum_order + binary * mask

        return sum_order

    # ---------------------------
    # Case 2: order >= 2
    # ---------------------------
    else:
        """
        Coupled two-expansion binarization:
          w ~ alpha1 * B1 + alpha2 * B2 + row_mean
        with alpha1, alpha2 solved in closed form 
        plus Tikhonov (ridge) stabilization term:
          + lam * (alpha1^2 + alpha2^2).
        """
        oc, ic = new_matrix.shape

        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                # If mask is all false in this row, skip
                continue

            row_vals = new_matrix[row_idx, row_mask]

            # (1) Subtract row mean
            row_mean = row_vals.mean()
            centered = row_vals - row_mean

            # (2) First expansion: B1, alpha1
            B1 = torch.sign(centered)
            alpha1 = centered.abs().mean()

            # (3) Second expansion: B2, alpha2 w.r.t. residual
            r = centered - alpha1 * B1
            B2 = torch.sign(r)
            alpha2 = r.abs().mean()

            # (4) Solve alpha1, alpha2 in closed form with Tikhonov
            d = float(row_vals.numel())
            c12 = torch.sum(B1 * B2).item()
            c1w = torch.sum(centered * B1).item()
            c2w = torch.sum(centered * B2).item()

            # The system is:
            #   [ (d + lam)   -c12     ] [ alpha1 ] = [ c1w ]
            #   [   -c12    (d + lam) ] [ alpha2 ]   [ c2w ]
            #
            # Denominator:
            denom = (d + lam) * (d + lam) - c12 * c12

            if abs(denom) > 1e-12:
                alpha1_new = ((d + lam) * c1w - c12 * c2w) / denom
                alpha2_new = ((d + lam) * c2w - c12 * c1w) / denom

                # Constrain to be non-negative
                alpha1 = max(alpha1_new, 0.0)
                alpha2 = max(alpha2_new, 0.0)
#            print(f"[ROW = {row_idx}] : alpha_1 : ", alpha1)
#            print(f"[ROW = {row_idx}] : alpha_2 : ", alpha2)

            # (5) Final approximation for that row
            approx_row = row_mean + alpha1 * B1 + alpha2 * B2
            sum_order[row_idx, row_mask] = approx_row

        return sum_order
@torch.no_grad()
def coupled_residual_binarization_stable_v2(
    x, 
    mask, 
    order=2, 
    lam=1e-5, 
    max_iters=3
):
    """
    A second version of the stabilized coupled binarization approach, 
    now with an iterative re-fitting step for the two-expansion case.

    Args:
      x (tensor): Weight matrix of shape (oc, ic).
      mask (tensor of bool): True where we binarize, False otherwise.
      order (int):
         - 1 => Single-pass binarization: w ~ alpha * sign(w - mean).
         - >= 2 => Two-expansion binarization with Tikhonov-stabilized 
                   alpha1, alpha2, plus iterative re-fitting of B2.
      lam (float): Tikhonov regularization term.
      max_iters (int): Number of small coordinate-descent iterations 
                       in the two-expansion step.

    Returns:
      sum_order (tensor): Binarized approximation of x.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    # ------------- CASE 1: order == 1 -------------
    if order == 1:
        # Single-pass binarization, exactly as before
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(
            mask, residual, 
            torch.tensor(float('nan'), device=residual.device)
        )

        # Row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Center each row
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )

        # Binarize
        binary = torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        # Add back the row mean
        binary += mean_tensor_all[:, None]

        sum_order += binary * mask
        return sum_order

    # ------------- CASE 2: order >= 2 -------------
    else:
        """
        Coupled 2-expansion binarization with Tikhonov regularization 
        + iterative B2 refitting.
        """
        oc, ic = new_matrix.shape

        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                # skip if this row is all unmasked
                continue

            row_vals = new_matrix[row_idx, row_mask]

            # 1) subtract mean
            row_mean = row_vals.mean()
            centered = row_vals - row_mean

            # 2) First expansion (B1, alpha1)
            B1 = torch.sign(centered)
            alpha1 = centered.abs().mean()

            # We'll run a small coordinate-descent loop
            # around (alpha1, alpha2, B2):
            alpha2 = 0.0
            B2 = torch.sign(centered)  # dummy init
            d = float(row_vals.numel())

            for _ in range(max_iters):
                # (a) Recompute residual after current alpha1,B1
                r = centered - alpha1 * B1 * 1.0  # copy
                # (b) Re-fit B2
                B2 = torch.sign(r)

                # (c) approximate alpha2 from the residual magnitude
                alpha2_guess = r.abs().mean()

                # (d) Solve alpha1, alpha2 in closed form with Tikhonov:
                c12 = torch.sum(B1 * B2).item()
                c1w = torch.sum(centered * B1).item()
                c2w = torch.sum(centered * B2).item()

                # Tikhonov system:
                #   [ (d + lam)  -c12      ] [ alpha1 ] = [ c1w ]
                #   [ -c12      (d + lam) ] [ alpha2 ]   [ c2w ]
                denom = (d + lam) * (d + lam) - (c12 * c12)

                if abs(denom) > 1e-12:
                    alpha1_new = ((d + lam)*c1w - c12 * c2w) / denom
                    alpha2_new = ((d + lam)*c2w - c12 * c1w) / denom
                    # clamp to nonnegative
                    alpha1 = max(alpha1_new, 0.0)
                    alpha2 = max(alpha2_new, 0.0)
                else:
                    # fallback
                    alpha1 = max(c1w / (d + lam), 0.0)
                    alpha2 = max(alpha2_guess, 0.0)

                # Optionally: if alpha2 is extremely small,
                # we might break early. But let's just keep
                # the iteration going to see if we can "revive"
                # alpha2 in the next pass. No early break.

            # done iteration

            # 3) final approximation for that row
            approx_row = row_mean + alpha1 * B1 + alpha2 * B2
            sum_order[row_idx, row_mask] = approx_row

        return sum_order
@torch.no_grad()
def coupled_residual_binarization_stable_v3(x, mask, order=2, lam=1e-5):
    """
    A single-pass, closed-form binarization with re-centering and Tikhonov stability.

    When order == 1:
      -> single alpha * sign( (w - row_mean) ).
    When order >= 2:
      -> alpha1 * B1 + alpha2 * B2 + row_mean, 
         with Tikhonov-stabilized closed-form for alpha1, alpha2
         AND a re-centering step for the second residual pass.

    Args:
      x (Tensor): the weight matrix of shape (oc, ic)
      mask (Tensor bool): which entries to binarize (True => use weight)
      order (int): 
         1 => single expansion, 
         >=2 => two expansions with stabilized coupling
      lam (float): Tikhonov (ridge) regularization parameter.

    Returns:
      sum_order (Tensor): approximate binarized reconstruction, same shape as x.
    """
    # We'll accumulate final approximation here
    sum_order = torch.zeros_like(x)
    # Copy & mask out invalid positions
    new_matrix = x.clone() * mask

    # optional: track usage count
    global index
    index += 1

    if order == 1:
        # ----------------------
        # Single-pass binarization
        # ----------------------
        residual = new_matrix  # or new_matrix - sum_order, but sum_order is 0
        masked_x_tensor = torch.where(
            mask, 
            residual, 
            torch.tensor(float('nan'), device=residual.device)
        )

        # Row mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Subtract row-wise mean
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale = average absolute value
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )

        # Binary sign
        binary = torch.sign(masked_x_tensor)
        # Multiply by scale
        binary *= scale_tensor_all[:, None]
        # Add row mean
        binary += mean_tensor_all[:, None]

        # Done
        sum_order = sum_order + binary * mask
        return sum_order

    else:
        # ----------------------
        # Two-expansion binarization with:
        #  (1) re-centering each pass 
        #  (2) Tikhonov for alpha1, alpha2
        # ----------------------
        oc, ic = new_matrix.shape

        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                # skip if no valid positions in this row
                continue

            row_vals = new_matrix[row_idx, row_mask]
            d = float(row_vals.numel())

            # (1) Row mean
            row_mean = row_vals.mean()  # not masked_x_tensor => same effect
            centered = row_vals - row_mean

            # (2) B1, alpha1 from the centered row
            B1 = torch.sign(centered)
            alpha1 = centered.abs().mean()

            # (3) Residual r
            r = centered - alpha1 * B1

            # (3a) Re-center r => reduce correlation
            r_mean = r.mean()
            r_centered = r - r_mean

            # B2, alpha2 from the re-centered residual
            B2 = torch.sign(r_centered)
            alpha2 = r_centered.abs().mean()

            # (4) Solve alpha1, alpha2 in one shot with Tikhonov
            #     Minimizing ||(w-mean) - alpha1 B1 - alpha2 B2||^2 + lam (alpha1^2 + alpha2^2).
            c12 = torch.sum(B1 * B2).item()
            c1w = torch.sum(centered * B1).item()  # <(w-mean), B1>
            c2w = torch.sum(centered * B2).item()  # <(w-mean), B2>

            # The system is:
            #   [d + lam  , -c12      ] [alpha1] = [ c1w ]
            #   [-c12     , d + lam   ] [alpha2]   [ c2w ]
            #
            denom = (d + lam) * (d + lam) - (c12 ** 2)
            if abs(denom) > 1e-12:
                alpha1_new = ((d + lam) * c1w - c12 * c2w) / denom
                alpha2_new = ((d + lam) * c2w - c12 * c1w) / denom

                # clamp to non-negative
                alpha1 = max(alpha1_new, 0.0)
                alpha2 = max(alpha2_new, 0.0)
            else:
                # fallback if denom is ~0
                # keep the naive alpha1, alpha2 from above
                pass

            # (5) Final row reconstruction
            # w_approx = row_mean + alpha1*B1 + alpha2*B2
            row_approx = row_mean + alpha1 * B1 + alpha2 * B2

            # place into sum_order
            sum_order[row_idx, row_mask] = row_approx

        return sum_order

import torch

@torch.no_grad()
def coupled_residual_binarization_stable_v4(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1
):
    """
    A single-pass, closed-form 2-expansion binarization with:
      - Re-centering of the row (for B1)
      - Re-centering of residual (for B2)
      - Tikhonov (ridge) regularization for stability
      - Correlation damping to avoid alpha2 => 0 if B1,B2 are strongly correlated

    When order == 1:
      -> single expansion:  w ~ alpha * sign( (w-row_mean) )
    When order >= 2:
      -> w ~ row_mean + alpha1 * B1 + alpha2 * B2
         with ridge-stabilized solution for alpha1, alpha2
         plus correlation damping on c12 if c12>0.

    Args:
      x (Tensor): (oc, ic) weight matrix
      mask (Bool Tensor): same shape as x, True => valid entries
      order (int): 
         - 1 => single expansion
         - >=2 => two expansions w/ correlation damping
      lam (float): Tikhonov/ridge strength
      corr_damp (float): factor in [0,1], how much to scale down positive c12

    Returns:
      sum_order (Tensor): approximate binarized reconstruction
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    if order == 1:
        # ---------------------------
        # Single-pass binarization
        # ---------------------------
        residual = new_matrix
        # Only valid positions
        masked_x_tensor = torch.where(
            mask,
            residual,
            torch.tensor(float('nan'), device=residual.device)
        )

        # Row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Subtract row mean
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale = avg abs value
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )

        # Binary sign
        binary = torch.sign(masked_x_tensor)
        # Scale
        binary *= scale_tensor_all[:, None]
        # Add row mean
        binary += mean_tensor_all[:, None]

        sum_order = sum_order + binary * mask
        return sum_order

    else:
        # ---------------------------
        # Two-expansion binarization
        # with re-centering + Tikhonov + correlation damping
        # ---------------------------
        oc, ic = new_matrix.shape

        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                continue

            row_vals = new_matrix[row_idx, row_mask]
            d = float(row_vals.numel())

            # 1) Row mean
            row_mean = row_vals.mean()
            centered = row_vals - row_mean

            # 2) B1, alpha1 from centered
            B1 = torch.sign(centered)
            alpha1 = centered.abs().mean()

            # 3) Residual r
            r = centered - alpha1 * B1

            # 3a) Re-center the residual
            r_mean = r.mean()
            r_centered = r - r_mean

            # B2, alpha2 from r_centered
            B2 = torch.sign(r_centered)
            alpha2 = r_centered.abs().mean()

            # 4) Tikhonov-stabilized closed-form for alpha1, alpha2
            c12 = torch.sum(B1 * B2).item()
            c1w = torch.sum(centered * B1).item()  # <(w-mean), B1>
            c2w = torch.sum(centered * B2).item()  # <(w-mean), B2>

            # Correlation damping if c12>0
            if c12 > 0:
                c12 = c12 * (1.0 - corr_damp)

            # Solve system:
            #   [d + lam, -c12   ] [alpha1] = [c1w]
            #   [-c12,   d + lam ] [alpha2]   [c2w]
            denom = (d + lam) * (d + lam) - (c12**2)
            if abs(denom) > 1e-12:
                alpha1_new = ((d + lam) * c1w - c12 * c2w) / denom
                alpha2_new = ((d + lam) * c2w - c12 * c1w) / denom
                # clamp non-negative
                alpha1 = max(alpha1_new, 0.0)
                alpha2 = max(alpha2_new, 0.0)
            else:
                # fallback if near-singular
                pass

            # 5) Final row approximation
            row_approx = row_mean + alpha1 * B1 + alpha2 * B2
            sum_order[row_idx, row_mask] = row_approx

        return sum_order

@torch.no_grad()
def coupled_residual_binarization_stable_v5(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1
):
    """
    A single-pass or two-pass binarization with various stabilizations.
    When order == 1: single expansion  w ~ row_mean + alpha * sign(w - row_mean)
                     now with Tikhonov regularization for alpha.
    When order >= 2: two expansions   w ~ row_mean + alpha1 B1 + alpha2 B2
                     with ridge-stabilized solution + correlation damping.
    ...
    """

    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    if order == 1:
        # ---------------------------
        # Single-pass binarization
        # with Tikhonov for alpha
        # ---------------------------
        residual = new_matrix

        masked_x_tensor = torch.where(
            mask,
            residual,
            torch.tensor(float('nan'), device=residual.device)
        )

        # Row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Subtract row mean
        centered_x = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale WITHOUT Tikhonov would be:
        #   scale_tensor_all = torch.nanmean(torch.abs(centered_x), dim=1)
        # Instead we do Tikhonov ridge:
        #   alpha = (||centered_x||_1) / (d + lam)
        # where d is the count of valid entries.
        abs_centered = torch.abs(centered_x)
        # number of valid entries in each row:
        valid_counts = torch.sum(~torch.isnan(centered_x), dim=1).float()
        l1_sums = torch.nan_to_num(abs_centered, 0.0).sum(dim=1)  # sum of absolute

        # Tikhonov scale
        # NOTE: we clamp at 1e-12 to avoid division by zero if no valid entries
        denom = valid_counts + lam
        denom = torch.where(denom < 1e-12, torch.tensor(1e-12, device=denom.device), denom)
        scale_tensor_all = l1_sums / denom

        # Binary sign
        binary = torch.sign(centered_x)

        # Multiply by alpha
        binary *= scale_tensor_all[:, None]

        # Add row mean
        binary += mean_tensor_all[:, None]

        # Final
        sum_order = sum_order + torch.where(torch.isnan(masked_x_tensor),
                                            torch.zeros_like(binary),
                                            binary)
        return sum_order

    else:
        # ---------------------------
        # Two-expansion binarization
        # with re-centering + Tikhonov + correlation damping
        # ---------------------------
        oc, ic = new_matrix.shape

        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                continue

            row_vals = new_matrix[row_idx, row_mask]
            d = float(row_vals.numel())
            if d < 1e-12:
                continue

            # 1) Row mean
            row_mean = row_vals.mean()
            centered = row_vals - row_mean

            # 2) B1, alpha1 from centered
            B1 = torch.sign(centered)
            alpha1 = centered.abs().mean()

            # 3) Residual r
            r = centered - alpha1 * B1
            # 3a) Re-center the residual
            r_mean = r.mean()
            r_centered = r - r_mean

            # B2, alpha2 from r_centered
            B2 = torch.sign(r_centered)
            alpha2 = r_centered.abs().mean()

            # 4) Tikhonov-stabilized closed-form for alpha1, alpha2
            c12 = torch.sum(B1 * B2).item()
            c1w = torch.sum(centered * B1).item()  # <(w-mean), B1>
            c2w = torch.sum(centered * B2).item()  # <(w-mean), B2>

            # Correlation damping if c12>0
            if c12 > 0:
                c12 = c12 * (1.0 - corr_damp)

            # Solve system:
            #   [d + lam,   -c12     ] [alpha1] = [c1w]
            #   [  -c12,    d + lam ] [alpha2]   [c2w]
            denom = (d + lam) * (d + lam) - (c12**2)
            if abs(denom) > 1e-12:
                alpha1_new = ((d + lam) * c1w - c12 * c2w) / denom
                alpha2_new = ((d + lam) * c2w - c12 * c1w) / denom
                alpha1 = max(alpha1_new, 0.0)
                alpha2 = max(alpha2_new, 0.0)

            # 5) Final row approximation
            row_approx = row_mean + alpha1 * B1 + alpha2 * B2
            sum_order[row_idx, row_mask] = row_approx

        return sum_order
@torch.no_grad()
def coupled_residual_binarization_stable_v6(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1
):
    """
    A single-pass (order==1) or two-expansion (order>=2) binarization with:
      - Row mean centering
      - Residual re-centering
      - Tikhonov (ridge) regularization
      - Correlation damping
      - *Sign refinement step* (new in v5) for B2 in two-expansion mode

    When order == 1:
      -> Single expansion: w ~ alpha * sign( (w - row_mean) )

    When order >= 2:
      -> w ~ row_mean + alpha1 * B1 + alpha2 * B2
         *with one sign-refinement pass* for B2 after solving alpha1, alpha2.

    Args:
      x (Tensor):         (oc, ic) weight matrix
      mask (Bool Tensor): same shape as x; True => valid entries
      order (int):        1 => single expansion, >=2 => two expansions
      lam (float):        Tikhonov/ridge strength
      corr_damp (float):  factor in [0,1], how much to scale down c12 if c12>0

    Returns:
      sum_order (Tensor): approximate binarized reconstruction, same shape as x
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    # ---------------------------
    # Case 1: single expansion
    # ---------------------------
    if order == 1:
        residual = new_matrix
        masked_x_tensor = torch.where(
            mask,
            residual,
            torch.tensor(float('nan'), device=residual.device)
        )

        # Row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Subtract row mean
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale = average absolute value
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )

        # Binary sign
        binary = torch.sign(masked_x_tensor)

        # Scale
        binary *= scale_tensor_all[:, None]

        # Add row mean
        binary += mean_tensor_all[:, None]

        sum_order = sum_order + binary * mask
        return sum_order

    # ---------------------------
    # Case 2: two expansions
    # with sign refinement (v5)
    # ---------------------------
    oc, ic = new_matrix.shape

    for row_idx in range(oc):
        row_mask = mask[row_idx, :]
        if not torch.any(row_mask):
            continue

        row_vals = new_matrix[row_idx, row_mask]
        d = float(row_vals.numel())

        # 1) Row mean
        row_mean = row_vals.mean()
        centered = row_vals - row_mean

        # 2) B1, alpha1 from centered
        B1 = torch.sign(centered)
        alpha1 = centered.abs().mean()

        # 3) Residual r
        r = centered - alpha1 * B1

        # 3a) Re-center the residual
        r_mean = r.mean()
        r_centered = r - r_mean

        # B2, alpha2 from r_centered
        B2 = torch.sign(r_centered)
        alpha2 = r_centered.abs().mean()

        # 4) Tikhonov-stabilized closed-form for alpha1, alpha2
        #    with correlation damping if c12>0
        def solve_alphas(B1, B2, c1w, c2w, d, lam, corr_damp):
            c12 = (B1 * B2).sum().item()
            if c12 > 0:
                c12 *= (1.0 - corr_damp)
            # System:
            #   [d + lam, -c12   ] [alpha1] = [c1w]
            #   [-c12,   d + lam ] [alpha2]   [c2w]
            denom = (d + lam) * (d + lam) - c12 * c12
            if abs(denom) > 1e-12:
                alpha1_new = ((d + lam) * c1w - c12 * c2w) / denom
                alpha2_new = ((d + lam) * c2w - c12 * c1w) / denom
                return max(alpha1_new, 0.0), max(alpha2_new, 0.0)
            else:
                # fallback if near-singular
                return 0.0, 0.0

        c1w = (centered * B1).sum().item()   # <(w-mean), B1>
        c2w = (centered * B2).sum().item()   # <(w-mean), B2>
        alpha1, alpha2 = solve_alphas(B1, B2, c1w, c2w, d, lam, corr_damp)

        # -------------------------
        # (NEW in v5) Sign-Refinement Step:
        # After alpha1, alpha2 are updated, recompute B2 from the
        # *actual* final residual (w-mean - alpha1*B1). Then re-solve.
        # -------------------------
        refined_residual = centered - alpha1 * B1
        # We do not re-add r_mean here; we've effectively folded that in
        # (since alpha1, alpha2 had already accounted for it).
        B2 = torch.sign(refined_residual)
        c2w_refined = (centered * B2).sum().item()  # updated <(w-mean), B2>
        alpha1, alpha2 = solve_alphas(B1, B2, c1w, c2w_refined, d, lam, corr_damp)

        # 5) Final row approximation
        row_approx = row_mean + alpha1 * B1 + alpha2 * B2
        sum_order[row_idx, row_mask] = row_approx

    return sum_order

@torch.no_grad()
def coupled_residual_binarization_stable_v7(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1
):
    print(corr_damp,lam)
    """
    A single-pass (order==1) or two-expansion (order>=2) binarization with:
      - Row mean centering
      - Residual re-centering
      - Tikhonov (ridge) regularization
      - Correlation damping
      - Two-way sign refinement (new in v7)

    For two expansions, the steps are:
      1) Solve alpha1, alpha2 for initial B1, B2
      2) Refine B2 => re-solve alpha1, alpha2
      3) Refine B1 => re-solve alpha1, alpha2

    This short coordinate-descent loop typically reduces the final error
    more robustly than v4/v5, with minimal extra cost.

    Args:
      x (Tensor):         (oc, ic) weight matrix
      mask (Bool Tensor): same shape as x; True => valid entries
      order (int):        1 => single expansion, >=2 => two expansions
      lam (float):        Tikhonov/ridge strength
      corr_damp (float):  factor in [0,1], how much to scale down c12 if c12>0

    Returns:
      sum_order (Tensor): approximate binarized reconstruction, same shape
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    # ---------------------------
    # Case 1: single expansion
    # ---------------------------
    if order == 1:
        residual = new_matrix
        masked_x_tensor = torch.where(
            mask,
            residual,
            torch.tensor(float('nan'), device=residual.device)
        )

        # Row-wise mean
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )

        # Subtract row mean
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]

        # Row-wise scale = average absolute value
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )

        # Binary sign
        binary = torch.sign(masked_x_tensor)

        # Scale
        binary *= scale_tensor_all[:, None]

        # Add row mean
        binary += mean_tensor_all[:, None]

        sum_order = sum_order + binary * mask
        return sum_order

    # ---------------------------
    # Case 2: two expansions
    # with two-way sign refinement (v7)
    # ---------------------------
    oc, ic = new_matrix.shape

    def solve_alphas(B1, B2, c1w, c2w, d, lam, corr_damp):
        # correlation
        c12 = (B1 * B2).sum().item()
        # if c12 is positive, damp it
        if c12 > 0:
            c12 *= (1.0 - corr_damp)

        # Solve system:
        #   [d + lam, -c12   ] [alpha1] = [c1w]
        #   [-c12,   d + lam ] [alpha2]   [c2w]
        denom = (d + lam) * (d + lam) - c12 * c12
        if abs(denom) > 1e-12:
            alpha1_new = ((d + lam) * c1w - c12 * c2w) / denom
            alpha2_new = ((d + lam) * c2w - c12 * c1w) / denom
            return max(alpha1_new, 0.0), max(alpha2_new, 0.0)
        else:
            # fallback if near-singular
            return 0.0, 0.0

    for row_idx in range(oc):
        row_mask = mask[row_idx, :]
        if not torch.any(row_mask):
            continue

        row_vals = new_matrix[row_idx, row_mask]
        d = float(row_vals.numel())

        # 1) Row mean
        row_mean = row_vals.mean()
        centered = row_vals - row_mean

        # 2) First expansion: B1, alpha1
        B1 = torch.sign(centered)
        alpha1 = centered.abs().mean()

        # 3) Residual => B2
        r = centered - alpha1 * B1
        r_mean = r.mean()
        r_centered = r - r_mean
        B2 = torch.sign(r_centered)
        alpha2 = r_centered.abs().mean()

        # 4) Solve alpha1, alpha2 (initial)
        c1w = (centered * B1).sum().item()   # <(w-mean), B1>
        c2w = (centered * B2).sum().item()   # <(w-mean), B2>
        alpha1, alpha2 = solve_alphas(B1, B2, c1w, c2w, d, lam, corr_damp)

        # 5) Recompute B2 => re-solve alpha1, alpha2
        B2 = torch.sign(centered - alpha1 * B1)
        c2w = (centered * B2).sum().item()
        alpha1, alpha2 = solve_alphas(B1, B2, c1w, c2w, d, lam, corr_damp)

        # (NEW in v7) 6) Recompute B1 => re-solve alpha1, alpha2
        B1 = torch.sign(centered - alpha2 * B2)
        c1w = (centered * B1).sum().item()
        alpha1, alpha2 = solve_alphas(B1, B2, c1w, c2w, d, lam, corr_damp)

        # 7) Final reconstruction
        row_approx = row_mean + alpha1 * B1 + alpha2 * B2
        sum_order[row_idx, row_mask] = row_approx

    return sum_order

@torch.no_grad()
def coupled_residual_binarization_stable_v8(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1
):
    """
    'v8' extends the two-expansion approach by also re-optimizing the row offset
    mu in the coordinate-descent loop. This yields a more accurate final
    approximation without adding any new bits or parameters (the offset is
    the same single row-mean float we already had).

    We keep:
      - Tikhonov (ridge) stabilization
      - Correlation damping
      - Two-way sign refinement for B1, B2
      - New: offset (mu) re-solved in closed-form.

    Args:
      x (Tensor):         (oc, ic) weight matrix
      mask (Bool Tensor): same shape as x, True => valid entries
      order (int):        1 => single expansion, >=2 => two expansions
      lam (float):        Tikhonov/ridge strength
      corr_damp (float):  factor in [0,1], how much to scale down positive c12

    Returns:
      sum_order (Tensor): approximate binarized reconstruction
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    # ---------------------------
    # Single expansion (unchanged)
    # ---------------------------
    if order == 1:
        residual = new_matrix
        masked_x_tensor = torch.where(
            mask,
            residual,
            torch.tensor(float('nan'), device=residual.device)
        )
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all
        )
        masked_x_tensor = masked_x_tensor - mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all
        )
        binary = torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary * mask
        return sum_order

    # --------------------------------------
    # Two expansions + offset refinement (v8)
    # --------------------------------------
    oc, ic = new_matrix.shape

    def solve_alphas(B1, B2, w_centered, d, lam, corr_damp):
        # correlation
        c12 = (B1 * B2).sum().item()
        # if c12 > 0, damp it
        if c12 > 0:
            c12 *= (1.0 - corr_damp)

        # <(w - mu), B1>, <(w - mu), B2>
        c1w = (w_centered * B1).sum().item()
        c2w = (w_centered * B2).sum().item()

        # Solve system:
        #   [d + lam, -c12   ] [alpha1] = [c1w]
        #   [-c12,   d + lam ] [alpha2]   [c2w]
        denom = (d + lam) * (d + lam) - c12 * c12
        if abs(denom) > 1e-12:
            alpha1_new = ((d + lam) * c1w - c12 * c2w) / denom
            alpha2_new = ((d + lam) * c2w - c12 * c1w) / denom
            alpha1_new = max(alpha1_new, 0.0)
            alpha2_new = max(alpha2_new, 0.0)
            return alpha1_new, alpha2_new
        else:
            return 0.0, 0.0

    for row_idx in range(oc):
        row_mask = mask[row_idx, :]
        if not torch.any(row_mask):
            continue

        row_vals = new_matrix[row_idx, row_mask]
        d = float(row_vals.numel())

        # 1) Initial offset = row mean
        mu = row_vals.mean()
        w_centered = row_vals - mu

        # 2) B1, alpha1 from w_centered
        B1 = torch.sign(w_centered)
        alpha1 = w_centered.abs().mean()

        # 3) B2 from residual (re-centered)
        r1 = w_centered - alpha1 * B1
        r1_mean = r1.mean()
        B2 = torch.sign(r1 - r1_mean)
        alpha2 = (r1 - r1_mean).abs().mean()

        # 4) Solve alpha1, alpha2 (initial)
        alpha1, alpha2 = solve_alphas(B1, B2, w_centered, d, lam, corr_damp)

        # 5) Sign refinement for B2
        B2 = torch.sign(w_centered - alpha1 * B1)
        alpha1, alpha2 = solve_alphas(B1, B2, w_centered, d, lam, corr_damp)

        # 6) Sign refinement for B1 (two-way)
        B1 = torch.sign(w_centered - alpha2 * B2)
        alpha1, alpha2 = solve_alphas(B1, B2, w_centered, d, lam, corr_damp)

        # 7) (NEW in v8) Refine offset mu, then re-solve alpha + sign
        #    a) mu = mean( w_i - alpha1 B_{1i} - alpha2 B_{2i} )
        w_res = row_vals - alpha1 * B1 - alpha2 * B2
        new_mu = w_res.mean()
        # b) Re-center for alpha solves
        w_centered = row_vals - new_mu

        # Re-solve alpha1, alpha2 with updated mu
        alpha1, alpha2 = solve_alphas(B1, B2, w_centered, d, lam, corr_damp)
        
        # c) Optional final sign refinements 
        B2 = torch.sign(w_centered - alpha1 * B1)
        alpha1, alpha2 = solve_alphas(B1, B2, w_centered, d, lam, corr_damp)
        B1 = torch.sign(w_centered - alpha2 * B2)
        alpha1, alpha2 = solve_alphas(B1, B2, w_centered, d, lam, corr_damp)

        # 8) Final reconstruction for that row
        row_approx = new_mu + alpha1 * B1 + alpha2 * B2
        sum_order[row_idx, row_mask] = row_approx

    return sum_order

@torch.no_grad()
def adaptive_high_order_residual(x, mask, order=2):
    """
    Adaptive High Order Residual Binarization.
    
    This function approximates the input tensor x (after applying mask)
    as a sum of binary components over the specified number of orders.
    
    For each order and for each channel, it computes two candidate scale factors:
    
      - Candidate 1: Uses the mean absolute deviation of the channel’s residual 
                     (as in braq), i.e. α₁ = nanmean(|r - m|).
      
      - Candidate 2: Uses the variance-based estimator, i.e. 
                     α₂ = sqrt(nanmean((r - m)²)) * sqrt(2/π),
                     which is optimal if the residual is Gaussian.
                     
    For each channel the candidate that yields the lower reconstruction error 
    is chosen adaptively. This approach minimizes quantization error under 
    non-ideal residual distributions while introducing no extra bits (the only 
    stored per-channel parameter remains the scale factor).
    
    When order = 1, the weights are represented as W ≈ α * B,
    and when order = 2, W ≈ α₁ * B₁ + α₂ * B₂, as required.
    
    Parameters:
      x (torch.Tensor): The weight tensor.
      mask (torch.Tensor): A binary mask of the same shape as x.
      order (int): The number of residual passes (default 2).
    
    Returns:
      torch.Tensor: The binarized approximation of x.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask
    global index
    index += 1
    # Prepare a NaN tensor for masked-out positions (ensuring device/dtype consistency)
    nan_tensor = torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
    
    for od in range(order):
        # Compute the current residual
        residual = new_matrix - sum_order
        # Apply the mask: invalid positions become NaN
        masked_x_tensor = torch.where(mask, residual, nan_tensor)
        
        # Compute channel-wise mean (serves as bias compensation)
        channel_mean = torch.nanmean(masked_x_tensor, dim=1)
        channel_mean = torch.where(torch.isnan(channel_mean), torch.zeros_like(channel_mean), channel_mean)
        
        # Center the residual by subtracting the channel mean
        centered = masked_x_tensor - channel_mean[:, None]
        
        # Candidate 1: Scale via mean absolute deviation (as in braq)
        candidate_scale1 = torch.nanmean(torch.abs(centered), dim=1)
        candidate_scale1 = torch.where(torch.isnan(candidate_scale1), torch.zeros_like(candidate_scale1), candidate_scale1)
        
        # Candidate 2: Scale via variance; for a Gaussian, E[|x|] = std * sqrt(2/π)
        candidate_std = torch.sqrt(torch.nanmean(centered**2, dim=1))
        candidate_scale2 = candidate_std * math.sqrt(2/math.pi)
        candidate_scale2 = torch.where(torch.isnan(candidate_scale2), torch.zeros_like(candidate_scale2), candidate_scale2)
        
        # Both candidates use the same binary pattern (the sign of the centered residual)
        binary = torch.sign(centered)
        
        # Reconstruct the candidate approximations
        rec1 = channel_mean[:, None] + candidate_scale1[:, None] * binary
        rec2 = channel_mean[:, None] + candidate_scale2[:, None] * binary
        
        # Compute per-channel reconstruction errors for both candidates
        error1 = torch.nanmean((masked_x_tensor - rec1)**2, dim=1)
        error2 = torch.nanmean((masked_x_tensor - rec2)**2, dim=1)
        
        # Select the candidate with lower error for each channel
        choose_candidate1 = (error1 <= error2)
        final_scale = torch.where(choose_candidate1, candidate_scale1, candidate_scale2)
        
        # Compute the final binary component for this iteration
        final_component = channel_mean[:, None] + final_scale[:, None] * binary
        
        # Update the accumulated representation (note: multiplication by mask preserves original sparsity)
        sum_order = sum_order + final_component * mask
        
    return sum_order

@torch.no_grad()
def adaptive_high_order_residual_v2(x, mask, order=2):
    """
    Adaptive High Order Residual Binarization using candidate 2 only.

    This function approximates the input tensor x (after applying mask)
    as a sum of binary components over the specified number of orders.
    Instead of adaptively choosing between two candidates, it always uses
    candidate 2, which computes the scale factor based on the variance:
      candidate_scale2 = sqrt(nanmean((r - m)**2)) * sqrt(2/π)
    where r is the residual and m is the channel-wise mean.

    Parameters:
      x (torch.Tensor): The weight tensor.
      mask (torch.Tensor): A binary mask of the same shape as x.
      order (int): The number of residual passes (default 2).

    Returns:
      torch.Tensor: The binarized approximation of x.
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask
    global index
    index += 1
    # Create a NaN tensor for masked-out positions (ensuring device/dtype consistency)
    nan_tensor = torch.tensor(float('nan'), device=x.device, dtype=x.dtype)

    for od in range(order):
        # Compute the current residual and apply the mask
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, nan_tensor)

        # Compute channel-wise mean for bias compensation
        channel_mean = torch.nanmean(masked_x_tensor, dim=1)
        channel_mean = torch.where(torch.isnan(channel_mean), torch.zeros_like(channel_mean), channel_mean)

        # Center the residual by subtracting the channel mean
        centered = masked_x_tensor - channel_mean[:, None]

        # Candidate 2: Scale via variance-based estimator (optimal if residual is Gaussian)
        candidate_std = torch.sqrt(torch.nanmean(centered**2, dim=1))
        candidate_scale2 = candidate_std * math.sqrt(2 / math.pi)
        candidate_scale2 = torch.where(torch.isnan(candidate_scale2), torch.zeros_like(candidate_scale2), candidate_scale2)

        # Use the sign of the centered residual as the binary component
        binary = torch.sign(centered)

        # Compute the final component using candidate 2's scale factor
        final_component = channel_mean[:, None] + candidate_scale2[:, None] * binary

        # Update the accumulated representation (multiplication by mask preserves sparsity)
        sum_order = sum_order + final_component * mask

    return sum_order

@torch.no_grad()
def hybrid_coupled_coordinate_residual(x, mask, order=2, lam=1e-5, corr_damp=0.1):
    """
    Hybrid binarization: runs both the stable coupled two-expansion method and braq,
    then selects the one with the lower quantization error.
    
    For order == 1 (non-salient regions), both methods reduce to a single expansion,
    while for order >= 2 (salient regions), the stable method applies re-centering,
    Tikhonov stabilization, and correlation damping.
    
    Args:
      x (Tensor): (oc, ic) weight matrix.
      mask (Bool Tensor): same shape as x; True indicates valid entries.
      order (int): 
         - 1 => single expansion: w ~ alpha * sign(w - row_mean)
         - >=2 => two expansions: w ~ row_mean + alpha1 * B1 + alpha2 * B2.
      lam (float): Tikhonov (ridge) regularization strength for stability.
      corr_damp (float): Factor in [0,1] to damp positive correlations (c12).
      
    Returns:
      Tensor: The approximate binarized reconstruction chosen from the method
              with the lower quantization error (squared L2 norm over valid entries).
    """
    # Compute approximation using braq (original high_order_residual method)
    approx_braq = high_order_residual(x, mask, order=order)
    
    # Compute approximation using the stable coupled residual binarization v4
    approx_cabr = coupled_residual_binarization_stable_v7(x, mask, order=order, lam=lam, corr_damp=corr_damp)
    
    # Compute the squared quantization error only over valid (masked) entries.
    error_braq = torch.sum(((x - approx_braq) * mask) ** 2)
    error_cabr = torch.sum(((x - approx_cabr) * mask) ** 2)
    
    # Choose the method with the lower error.
    if error_braq <= error_cabr:
        return approx_braq
    else:
        return approx_cabr


@torch.no_grad()
def bit_flip_pass(w, mask, order=2):
    """
    Implements an order-aware bit-flipping binarization technique.
    
    w: (oc, ic) block of weights.
    mask: Boolean mask of valid entries.
    order: 1 for single-pass, 2 for residual-based refinement.
    
    Returns: The binarized weight matrix with optimized bit flips.
    Complexity: O(N).
    """
    # Order = 1 (direct binarization)
    active_w = w[mask]
    if active_w.numel() == 0:
        return torch.zeros_like(w)

    alpha_1 = active_w.abs().mean()
    B_1 = torch.sign(w) * mask
    R_1 = w - alpha_1 * B_1

    # Single-pass bit flipping for order=1
    for row_idx in range(w.shape[0]):
        row_mask = mask[row_idx]
        row_r = R_1[row_idx]
        row_b = B_1[row_idx]

        active_indices = torch.where(row_mask)[0]
        for col_idx in active_indices:
            if row_b[col_idx] > 0 and row_r[col_idx] < -alpha_1:
                row_b[col_idx] = -1.0
                row_r[col_idx] += 2.0 * alpha_1
            elif row_b[col_idx] < 0 and row_r[col_idx] > alpha_1:
                row_b[col_idx] = 1.0
                row_r[col_idx] -= 2.0 * alpha_1

        B_1[row_idx] = row_b
        R_1[row_idx] = row_r

    # If order = 1, return the refined first binarization
    if order == 1:
        return alpha_1 * B_1

    # Order = 2 (residual binarization)
    R_2 = w - alpha_1 * B_1  # First-order residual
    active_r = R_2[mask]
    alpha_2 = active_r.abs().mean() if active_r.numel() > 0 else 0.0

    B_2 = torch.sign(R_2) * mask
    R_2 -= alpha_2 * B_2  # New residual

    # Single-pass bit flipping for order=2
    for row_idx in range(w.shape[0]):
        row_mask = mask[row_idx]
        row_r = R_2[row_idx]
        row_b = B_2[row_idx]

        active_indices = torch.where(row_mask)[0]
        for col_idx in active_indices:
            if row_b[col_idx] > 0 and row_r[col_idx] < -alpha_2:
                row_b[col_idx] = -1.0
                row_r[col_idx] += 2.0 * alpha_2
            elif row_b[col_idx] < 0 and row_r[col_idx] > alpha_2:
                row_b[col_idx] = 1.0
                row_r[col_idx] -= 2.0 * alpha_2

        B_2[row_idx] = row_b
        R_2[row_idx] = row_r

    # Final binarized weight: sum of two binarized components
    return alpha_1 * B_1 + alpha_2 * B_2
@torch.no_grad()
def coupled_residual_binarization_stable_v9(
    x,
    mask,
    order=2,
    lam=1e-5,
    corr_damp=0.1
):
    """
    'v9' - A minimal single-pass approach that extends braq by a single
    closed-form coupling of alpha1, alpha2 after determining B1,B2.

    - If order=1: w ~ alpha * sign(w)
    - If order>=2: w ~ alpha1 * B1 + alpha2 * B2
      where B1=sign(w), B2=sign(r) with r=(w - alpha1^0*B1),
      then solve alpha1, alpha2 jointly in closed form with Tikhonov (lam)
      and optional correlation damping.

    No iterative refinement or row-mean shifting. This is simpler yet
    typically outperforms braq because alpha1 is re-optimized after
    seeing B2, rather than locked to alpha1^0.

    Args:
      x (Tensor):         (oc, ic) weight matrix
      mask (Bool Tensor): same shape as x
      order (int):        1 => single expansion, >=2 => two expansions
      lam   (float):      Tikhonov ridge for alpha1^2+alpha2^2
      corr_damp(float):   factor in [0,1]; scale down correlation if c12>0

    Returns:
      sum_order(Tensor):  same shape as x; final approximation
    """
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask

    global index
    index += 1

    if order == 1:
        # Single expansion: w ~ alpha * sign(w)
        # This is the same standard approach.
        masked_x = torch.where(mask, new_matrix, torch.tensor(float('nan'), device=x.device))
        scale = torch.nanmean(torch.abs(masked_x), dim=1)  # row-wise
        scale = torch.where(torch.isnan(scale), torch.zeros_like(scale), scale)
        sign_mat = torch.sign(masked_x)
        sign_mat *= scale[:, None]
        sum_order = sign_mat.where(mask, torch.zeros_like(sign_mat))
        return sum_order

    # Two expansions
    oc, ic = new_matrix.shape

    for row_idx in range(oc):
        row_mask = mask[row_idx, :]
        if not torch.any(row_mask):
            continue

        w_row = new_matrix[row_idx, row_mask]
        d = float(w_row.numel())

        # 1) B1 = sign(w), alpha1^0 = mean(|w|)
        B1 = torch.sign(w_row)
        alpha1_0 = w_row.abs().mean()

        # 2) Residual => B2 = sign(r)
        R = w_row - alpha1_0 * B1
        B2 = torch.sign(R)
        alpha2_0 = R.abs().mean()  # just for reference, not final

        # 3) Solve alpha1, alpha2 in one shot with Tikhonov + correlation damping
        c11 = d + lam  # effectively <B1,B1> + lam
        c22 = d + lam  # effectively <B2,B2> + lam
        c12 = (B1 * B2).sum().item()
        if c12 > 0:
            c12 *= (1.0 - corr_damp)
        c1w = (B1 * w_row).sum().item()
        c2w = (B2 * w_row).sum().item()

        # 2x2 system
        denom = c11 * c22 - c12 * c12
        if abs(denom) < 1e-12:
            # fallback
            alpha1, alpha2 = alpha1_0, alpha2_0
        else:
            alpha1 = ( c1w*c22 - c12*c2w ) / denom
            alpha2 = ( c2w*c11 - c12*c1w ) / denom
            alpha1 = max(alpha1, 0.0)
            alpha2 = max(alpha2, 0.0)

        # 4) Reconstruct final row
        #    w_approx = alpha1*B1 + alpha2*B2
        w_approx = alpha1 * B1 + alpha2 * B2
        sum_order[row_idx, row_mask] = w_approx

    return sum_order
@torch.no_grad()
def coupled_residual_binarization_stable_v10(
    x,
    mask,
    order=2,
    eps=1e-12
):
    """
    A streamlined binarization method with zero offsets and a
    single-pass approach for stability.

    If order=1:
      - w ~ alpha * sign(w), using average magnitude for alpha.

    If order>=2:
      - 1) B1=sign(w), alpha1=mean(|w|)
      - 2) r=w-alpha1*B1, B2=sign(r), alpha2=mean(|r|)
      - 3) 'Scale Sharing': let alphaTotal=mean(|w|) for that row,
         then rescale alpha1' and alpha2' so alpha1'+alpha2'=alphaTotal.

    This ensures neither alpha gets excessively large or vanishingly small,
    while remaining extremely simple (one pass, no offset, no iteration).

    Args:
      x (Tensor):         shape (oc, ic)
      mask (Bool Tensor): shape (oc, ic), True => valid entries
      order (int):        1 => single expansion, >=2 => two expansions
      eps (float):        small constant to avoid divisions by zero

    Returns:
      sum_order (Tensor): same shape as x, binarized approximation
    """
    sum_order = torch.zeros_like(x)
    # We'll operate on a masked copy
    new_matrix = x.clone() * mask

    # We'll do row-by-row. Each row we consider only the "True" positions in mask.
    oc, ic = new_matrix.shape

    if order == 1:
        # Single expansion
        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                continue

            row_vals = new_matrix[row_idx, row_mask]  # all valid positions
            # alpha = average magnitude
            alpha = row_vals.abs().mean()
            # sign
            B = torch.sign(row_vals)
            # final reconstruction
            row_approx = alpha * B
            sum_order[row_idx, row_mask] = row_approx

    else:
        # Two expansions + scale sharing
        for row_idx in range(oc):
            row_mask = mask[row_idx, :]
            if not torch.any(row_mask):
                continue

            row_vals = new_matrix[row_idx, row_mask]
            d = float(row_vals.numel())

            # Step A: B1, alpha1
            B1 = torch.sign(row_vals)
            alpha1 = row_vals.abs().mean()

            # Step B: Residual -> B2, alpha2
            r = row_vals - alpha1 * B1
            B2 = torch.sign(r)
            alpha2 = r.abs().mean()

            # Step C: scale sharing
            # total scale = average magnitude of original row
            alpha_total = row_vals.abs().mean()

            alpha_sum = alpha1 + alpha2
            if alpha_sum < eps:
                # edge case: if alpha1+alpha2 is basically 0, just do alpha_total for alpha1
                alpha1_prime = alpha_total
                alpha2_prime = 0.0
            else:
                alpha1_prime = alpha_total * (alpha1 / alpha_sum)
                alpha2_prime = alpha_total * (alpha2 / alpha_sum)

            # final reconstruction
            row_approx = alpha1_prime * B1 + alpha2_prime * B2
            sum_order[row_idx, row_mask] = row_approx

    return sum_order

class Binarization(nn.Module):
    def __init__(self, weight, method="2bit", groupsize=-1, corr_damp = 0.1, lam = 1e-5 ):
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

        self.corr_damp = corr_damp
        self.lam = lam

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
        elif self.method == 'crbog':
            w = coupled_residual_binarization(w, mask, order=order) 
        elif self.method=="crb":  # <-- NEW PROPOSAL
            w = coupled_residual_binarization_stable_v7(w, mask, order=order, corr_damp = self.corr_damp, lam = self.lam)
        elif self.method=="crbv8":  # <-- NEW PROPOSAL
            w = coupled_residual_binarization_stable_v8(w, mask, order=order)
        elif self.method=="crbv9":  # <-- NEW PROPOSAL
            w = coupled_residual_binarization_stable_v9(w, mask, order=order)
        elif self.method=="crbv10":  # <-- NEW PROPOSAL
            w = coupled_residual_binarization_stable_v10(w, mask, order=order)

        elif self.method=="new":  # <-- NEW PROPOSAL
            #w = hybrid_coupled_coordinate_residual(w, mask, order=order)
            w = bit_flip_pass(w, mask, order=order)
        elif self.method == 'ahor':
            w = adaptive_high_order_residual_v2(w,mask,order=order)
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
