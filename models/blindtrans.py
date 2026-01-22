import torch
import torch.nn as nn
import torch.nn.functional as F
from . import regist_model
from .mybi import PatchSelectiveTransformer
@regist_model
class TDBSNl(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9):
        super().__init__()
        assert base_ch % 2 == 0
        ly = []
        ly += [nn.Conv2d(in_ch, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.head = nn.Sequential(*ly)
        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = TransformerBranch(base_ch, base_ch, num_heads=4, depth=4)
        self.tpblock = TPBlock(channels=base_ch * 2)
        ly = []
        ly += [nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, out_ch, kernel_size=1)]
        self.tail = nn.Sequential(*ly)
    def forward(self, x):
        x = self.head(x)
        br1 = self.branch1(x)
        br2 = self.branch2(x)
        x = torch.cat([br1, br2], dim=1)
        x = self.tpblock(x)
        return self.tail(x)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()
        ly = []
        ly += [CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [DCl(stride, in_ch) for _ in range(num_module)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*ly)
    def forward(self, x):
        return self.body(x)
class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()
        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)
    def forward(self, x):
        return x + self.body(x)
class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PatchSelectiveTransformer(dim=dim, num_heads=num_heads, patch_size=16, top_k=4)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)),nn.GELU(),nn.Linear(int(dim * mlp_ratio), dim),nn.Dropout(dropout))
    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.attn(x_norm)
        x_norm = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm2(x_norm)
        x_mlp = self.mlp(x_norm)
        x_mlp = x_mlp.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + x_mlp
        return x
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class TransformerBranch(nn.Module):
    def __init__(self, in_ch, base_ch, num_heads=4, depth=4):
        super(TransformerBranch, self).__init__()
        self.blind_conv = CentralMaskedConv2d(in_ch, base_ch, kernel_size=5, padding=2)
        self.conv = nn.Sequential(nn.Conv2d(base_ch, base_ch, kernel_size=1),nn.ReLU(inplace=True))
        self.transformer = TransformerEncoder(base_ch//4, depth=depth, num_heads=num_heads)
        self.conv2 = nn.Sequential(nn.Conv2d(base_ch, base_ch//4, kernel_size=1),nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(base_ch//4, base_ch, kernel_size=1),nn.ReLU(inplace=True))
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.blind_conv(x)
        x = self.conv(x)
        x = self.conv2(x)
        x = self.transformer(x)
        x = self.conv3(x)
        return x
class TPBlock(nn.Module):
    def __init__(self, channels):
        super(TPBlock, self).__init__()
        self.channels = channels
        self.branch1_conv1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0)),nn.ReLU(inplace=True))
        self.branch1_conv2 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(2, 0), dilation=(2, 1)),nn.ReLU(inplace=True))
        self.branch2_conv1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1)),nn.ReLU(inplace=True))
        self.branch2_conv2 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2)),nn.ReLU(inplace=True))
        self.final_conv = nn.Conv2d(2 * channels, channels, kernel_size=1)
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x1 = x.mean(dim=3, keepdim=True)
        a = torch.sigmoid(x1)
        x1_weighted = x1 + x1 * a
        x1_conv1 = self.branch1_conv1(x1_weighted)
        x1_conv2 = self.branch1_conv2(x1_weighted)
        branch1 = torch.cat([x1_conv1, x1_conv2], dim=1)
        x2 = x.mean(dim=2, keepdim=True)
        b = torch.sigmoid(x2)
        x2_weighted = x2 + x2 * b
        x2_conv1 = self.branch2_conv1(x2_weighted)
        x2_conv2 = self.branch2_conv2(x2_weighted)
        branch2 = torch.cat([x2_conv1, x2_conv2], dim=1)
        branch1_expanded = branch1.expand(-1, -1, -1, width)
        branch2_expanded = branch2.expand(-1, -1, height, -1)
        fused = branch1_expanded * branch2_expanded
        y = self.final_conv(fused)
        c = torch.sigmoid(y)
        output = x + x * c
        return output