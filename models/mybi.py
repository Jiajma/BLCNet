import torch
import torch.nn as nn
class PatchSelectiveTransformer(nn.Module):
    def __init__(self, dim, num_heads, patch_size=16, top_k=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.top_k = top_k
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        num_patches_h, num_patches_w = H // P, W // P
        num_patches = num_patches_h * num_patches_w
        x_patches = x.view(B, C, num_patches_h, P, num_patches_w, P)
        x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        x_patches = x_patches.view(B, num_patches, C, P, P)
        x_flatten = x_patches.flatten(2)
        similarity = torch.matmul(x_flatten, x_flatten.transpose(1, 2))
        _, top_k_indices = torch.topk(similarity, k=self.top_k + 1, dim=-1)
        top_k_indices = top_k_indices[:, :, 1:]
        output_patches = []
        for row in range(num_patches_h):
            row_start = row * num_patches_w
            row_end = (row + 1) * num_patches_w
            current_row_indices = torch.arange(row_start, row_end, device=x.device)
            row_top_k = top_k_indices[:, current_row_indices, :]
            batch_indices = torch.arange(B, device=x.device)[:, None, None]
            similar_patches = x_patches[batch_indices, row_top_k]
            current_patches = x_patches[:, current_row_indices].unsqueeze(2)
            patches_to_attend = torch.cat([current_patches, similar_patches], dim=2)
            patches_to_attend = patches_to_attend.view(B * num_patches_w,(1 + self.top_k) * P * P,C)
            attended_patches, _ = self.attention(patches_to_attend, patches_to_attend, patches_to_attend)
            current_output = attended_patches[:, :P*P, :].view(B, num_patches_w, C, P, P)
            output_patches.append(current_output)
        output = torch.cat(output_patches, dim=1)
        output = output.permute(0, 2, 1, 3, 4).reshape(B, C, H, W)
        output = self.smooth_boundary(output, num_patches_h, num_patches_w, self.patch_size)
        return output
    def smooth_boundary(self, output_patches, num_patches_h, num_patches_w, patch_size):
        B, C, H, W = output_patches.shape
        P = patch_size
        output = output_patches.clone()
        if num_patches_h > 1:
            h_starts = torch.arange(1, num_patches_h, device=output_patches.device) * P
            output[:, :, h_starts, :] = (output_patches[:, :, h_starts, :] * 0.5+output_patches[:, :, h_starts - 1, :] * 0.5)
        if num_patches_w > 1:
            w_starts = torch.arange(1, num_patches_w, device=output_patches.device) * P
            output[:, :, :, w_starts] = (output_patches[:, :, :, w_starts] * 0.5+output_patches[:, :, :, w_starts - 1] * 0.5)
        return output