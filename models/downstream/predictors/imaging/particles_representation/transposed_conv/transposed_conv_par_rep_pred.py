from typing import Dict, Tuple

import numpy as np
from torch import nn
import torch.nn.functional as F
from models.downstream.predictors.imaging.image_pred import ImagePred
from models.downstream.utils import *


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class TransposedConvParRepPred(ImagePred):
    def __init__(self, config_path: str, representation_size: int):
        super().__init__(config_path, representation_size)
        self.image_predictor = self._build_image_predictor()

    def _build_image_predictor(self):
        """
        Builds the sequential model that maps [N_active, RepSize] -> [N_active, NumBins, PatchSize, PatchSize]
        """
        patch_size = self.config.particle_image_pred_size
        num_bins = self.config.num_bins
        add_alpha = self.config.add_alpha_channel
        out_channels = num_bins + 1 if add_alpha else num_bins
        initial_res = 4
        hidden_ch = self.config.num_channels

        num_upsamples = int(torch.log2(torch.tensor(patch_size / initial_res)))

        layers = []

        # 1. Projection and Reshape
        layers.append(nn.Linear(self.representation_size, hidden_ch * initial_res * initial_res))
        layers.append(Reshape(hidden_ch, initial_res, initial_res))
        layers.append(nn.ReLU())

        # 2. Upsampling Stack
        current_ch = hidden_ch
        for _ in range(num_upsamples):
            layers.append(nn.ConvTranspose2d(
                in_channels=current_ch,
                out_channels=current_ch // 2,
                kernel_size=4,
                stride=2,
                padding=1
            ))
            layers.append(nn.ReLU())
            current_ch //= 2

        # 3. Final refinement to num_bins channels
        layers.append(nn.Conv2d(current_ch, out_channels, kernel_size=3, padding=1))

        return nn.Sequential(*layers)
    def calculate_canvas_uncertainty(self, canvas: torch.Tensor) -> np.ndarray:
        """
        Calculates uncertainty map from the canvas by computing entropy over the bins.
        Args:
            canvas: [B, NumBins, H, W]
        Returns:
            uncertainty_map: [B, H, W] numpy array
        """
        with torch.no_grad():
            probs = torch.softmax(canvas, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [B, H, W]
        return entropy

    def inference(self, representation: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            representation: [B, G, D]
            kwargs:
                - particles_locations: [B, G, 2]
                - active_particles_mask: [B, G]
                - img_coords: [2, 2] Tensor. [[min_x, min_y], [max_x, max_y]]
                - unlimited_area: bool (optional, overrides img_coords if True)
        Returns:
            canvas: [B, NumBins, H, W]
            info: {"predicted_images": canvas}
        """
        # 1. Extract Active Particles
        active_mask = kwargs.get("active_particles_mask")  # [B, G]
        active_reps = representation[active_mask]  # [N_total, D]

        # 2. Forward Pass
        # patches shape: [N_total, OutChannels, S, S]
        patches = self.image_predictor(active_reps)

        # 3. Setup Canvas
        B, G = active_mask.shape
        H, W = self.config.image_size, self.config.image_size
        S = self.config.particle_image_pred_size
        num_bins = self.config.num_bins

        # Configs
        add_alpha = getattr(self.config, "add_alpha_channel", False)
        addition_mode = getattr(self.config, "particle_addition_mode", "sum")
        unlimited_area = getattr(self.config, "unlimited_area", False)

        particles_locations = kwargs.get("particles_locations")
        device = representation.device

        # Parse Viewport
        img_coords = kwargs.get("img_coords")
        ref_min_x = img_coords[0, 0].item()
        ref_max_x = img_coords[1, 0].item()
        ref_min_y = img_coords[0, 1].item()
        ref_max_y = img_coords[1, 1].item()

        # Calculate Base Resolution
        span_x = max(1e-6, ref_max_x - ref_min_x)
        span_y = max(1e-6, ref_max_y - ref_min_y)
        res_x = W / span_x
        res_y = H / span_y

        # --- INITIALIZATION ---
        # If using 'max' or 'most_certain', we init with -inf so patches overwrite background.
        # For 'mean' or 'sum', we init with 0.
        init_val = -float('inf') if addition_mode in ["max", "most_certain"] and not add_alpha else 0.0
        canvas = torch.full((B, num_bins, H, W), init_val, device=device)

        # Auxiliary buffers for specific modes (used only if NOT unlimited_area)
        # If unlimited_area is True, these are created dynamically per batch item.
        global_count_buffer = None
        global_conf_buffer = None

        if not unlimited_area and not add_alpha:
            if addition_mode == "mean":
                global_count_buffer = torch.zeros(B, 1, H, W, device=device)
            elif addition_mode == "most_certain":
                global_conf_buffer = torch.full((B, 1, H, W), -1.0, device=device)

        current_idx = 0

        for b in range(B):
            mask_b = active_mask[b]
            num_active = mask_b.sum().item()

            if num_active == 0:
                continue

            patches_b = patches[current_idx: current_idx + num_active]
            locs_b = particles_locations[b][mask_b]

            # --- DETERMINE VIEWPORT & TARGET BUFFERS ---
            if unlimited_area:
                # Dynamic sizing logic...
                p_min_vals, _ = locs_b.min(dim=0)
                p_max_vals, _ = locs_b.max(dim=0)

                pad_x = (S / 2.0) / res_x
                pad_y = (S / 2.0) / res_y

                curr_min_x = p_min_vals[0].item() - pad_x
                curr_max_x = p_max_vals[0].item() + pad_x
                curr_min_y = p_min_vals[1].item() - pad_y
                curr_max_y = p_max_vals[1].item() + pad_y

                temp_w = int((curr_max_x - curr_min_x) * res_x)
                temp_h = int((curr_max_y - curr_min_y) * res_y)

                temp_w = max(S, min(temp_w, 4096))
                temp_h = max(S, min(temp_h, 4096))

                # Create temp target canvas
                target_canvas = torch.full((1, num_bins, temp_h, temp_w), init_val, device=device)
                draw_w, draw_h = temp_w, temp_h

                # Create temp aux buffers
                count_buffer = torch.zeros(1, 1, temp_h, temp_w, device=device) if addition_mode == "mean" else None
                conf_buffer = torch.full((1, 1, temp_h, temp_w), -1.0,
                                         device=device) if addition_mode == "most_certain" else None

            else:
                # Fixed bounds
                curr_min_x, curr_max_x = ref_min_x, ref_max_x
                curr_min_y, curr_max_y = ref_min_y, ref_max_y

                # Use global buffers
                target_canvas = canvas[b].unsqueeze(0)
                draw_w, draw_h = W, H
                count_buffer = global_count_buffer[b].unsqueeze(0) if global_count_buffer is not None else None
                conf_buffer = global_conf_buffer[b].unsqueeze(0) if global_conf_buffer is not None else None

            # --- PROJECTION LOGIC ---
            curr_span_x = max(1e-6, curr_max_x - curr_min_x)
            curr_span_y = max(1e-6, curr_max_y - curr_min_y)

            norm_x = (locs_b[:, 0] - curr_min_x) / curr_span_x
            norm_y = (locs_b[:, 1] - curr_min_y) / curr_span_y

            center_x = norm_x * (draw_w - 1)
            center_y = norm_y * (draw_h - 1)

            tl_x = (center_x - S / 2).long()
            tl_y = (center_y - S / 2).long()

            # --- PASTE PATCHES ---
            for i in range(num_active):
                x_start = tl_x[i].item()
                y_start = tl_y[i].item()
                x_end = x_start + S
                y_end = y_start + S

                # Intersection
                c_x1, c_x2 = max(0, x_start), min(draw_w, x_end)
                c_y1, c_y2 = max(0, y_start), min(draw_h, y_end)

                if c_x1 >= c_x2 or c_y1 >= c_y2:
                    continue

                p_x1 = c_x1 - x_start
                p_x2 = c_x2 - x_start
                p_y1 = c_y1 - y_start
                p_y2 = c_y2 - y_start

                current_patch = patches_b[i, :, p_y1:p_y2, p_x1:p_x2]

                # -----------------------------------------------------
                # MODE SELECTION
                # -----------------------------------------------------
                if add_alpha:
                    # Alpha Blending (Precedes addition modes)
                    canvas_slice = target_canvas[0, :, c_y1:c_y2, c_x1:c_x2].clone()
                    p_content = current_patch[:num_bins]
                    p_alpha = torch.sigmoid(current_patch[num_bins:])
                    blended = (p_content * p_alpha) + (canvas_slice * (1.0 - p_alpha))
                    target_canvas[0, :, c_y1:c_y2, c_x1:c_x2] = blended

                elif addition_mode == "sum":
                    target_canvas[0, :, c_y1:c_y2, c_x1:c_x2] += current_patch

                elif addition_mode == "max":
                    # Take max per pixel
                    canvas_slice = target_canvas[0, :, c_y1:c_y2, c_x1:c_x2].clone()
                    target_canvas[0, :, c_y1:c_y2, c_x1:c_x2] = torch.max(canvas_slice, current_patch)

                elif addition_mode == "mean":
                    # Accumulate sum and count
                    target_canvas[0, :, c_y1:c_y2, c_x1:c_x2] += current_patch
                    count_buffer[0, :, c_y1:c_y2, c_x1:c_x2] += 1.0

                elif addition_mode == "most_certain":
                    # Calculate certainty: max prob from softmax over bins
                    # current_patch shape: [NumBins, H_slice, W_slice]
                    probs = torch.softmax(current_patch, dim=0)
                    certainty, _ = probs.max(dim=0, keepdim=True)  # [1, H_slice, W_slice]

                    current_conf_slice = conf_buffer[0, :, c_y1:c_y2, c_x1:c_x2]

                    # Mask where new patch is more certain than existing
                    update_mask = certainty > current_conf_slice

                    # Update canvas and buffer
                    # We expand mask to [NumBins, H, W] for the canvas
                    canvas_slice = target_canvas[0, :, c_y1:c_y2, c_x1:c_x2]
                    target_canvas[0, :, c_y1:c_y2, c_x1:c_x2] = torch.where(update_mask, current_patch, canvas_slice)

                    conf_buffer[0, :, c_y1:c_y2, c_x1:c_x2] = torch.where(update_mask, certainty, current_conf_slice)

            # --- FINALIZE BATCH ITEM ---
            if unlimited_area:
                # Apply Mean Division if needed before resizing
                if not add_alpha and addition_mode == "mean":
                    target_canvas = target_canvas / count_buffer.clamp(min=1.0)

                # Resize to final
                resized = F.interpolate(target_canvas, size=(H, W), mode='bilinear', align_corners=False)

                # If modes were max/most_certain, we might have -inf in the result (untouched areas).
                # We replace them with 0 (background)
                if not add_alpha and addition_mode in ["max", "most_certain"]:
                    resized = torch.nan_to_num(resized, neginf=0.0)

                canvas[b] = resized.squeeze(0)

            else:
                # Fixed Area Post-Processing
                # If mean, we do it at the very end outside loop?
                # No, we can do it here if we want per-batch correctness, but we updated canvas[b] in place.
                pass

            current_idx += num_active

        # --- FINAL CLEANUP (For Fixed Area) ---
        if not unlimited_area and not add_alpha:
            if addition_mode == "mean":
                canvas = canvas / global_count_buffer.clamp(min=1.0)
            elif addition_mode in ["max", "most_certain"]:
                # Convert remaining init -inf to 0
                canvas = torch.nan_to_num(canvas, neginf=0.0)

        return canvas, {"predicted_images": canvas, "canvas_uncertainty": self.calculate_canvas_uncertainty(canvas)}


