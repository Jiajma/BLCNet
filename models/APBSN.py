import torch
import torch.nn as nn
import torch.nn.functional as F
from .apbsn_util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
from . import regist_model
from .blindtrans import TDBSNl
@regist_model
class APBSN(nn.Module):
    def __init__(self, pd_a=4, pd_b=2, pd_pad=0, R3=True, R3_T=8, R3_p=0.16,
                    bsn='DBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9):
        super().__init__()
        self.pd_a    = pd_a
        self.pd_b    = pd_b
        self.pd_pad  = pd_pad
        self.R3      = R3
        self.R3_T    = R3_T
        self.R3_p    = R3_p
        if bsn == 'TDBSNl':
            self.bsn = TDBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
    def forward(self, img, pd=None):
        if pd is None: pd = self.pd_a
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            pd_img = F.pad(img, (p,p,p,p))
        pd_img_denoised = self.bsn(pd_img)
        if pd > 1:
            img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            img_pd_bsn = pd_img_denoised[:,:,p:-p,p:-p]
        return img_pd_bsn
    def denoise(self, x):
        b,c,h,w = x.shape
        if h % self.pd_b != 0:
            x = F.pad(x, (0, 0, 0, self.pd_b - h%self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            x = F.pad(x, (0, self.pd_b - w%self.pd_b, 0, 0), mode='constant', value=0)
        img_pd_bsn = self.forward(img=x, pd=self.pd_b)
        if not self.R3:
            return img_pd_bsn[:,:,:h,:w]
        else:
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p
                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = x[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p,p,p,p), mode='reflect')
                if self.pd_pad == 0:
                    denoised[..., t] = self.bsn(tmp_input)
                else:
                    denoised[..., t] = self.bsn(tmp_input)[:,:,p:-p,p:-p]
            return torch.mean(denoised, dim=-1)