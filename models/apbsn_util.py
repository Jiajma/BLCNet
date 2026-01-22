from math import exp
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
def np2tensor(n:np.array):
    if len(n.shape) == 2:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(n, (2,0,1))))
    elif len(n.shape) == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.flip(n, axis=2), (2,0,1))))
def tensor2np(t:torch.Tensor):
    t = t.cpu().detach()
    if len(t.shape) == 2:
        return t.permute(1,2,0).numpy()
    elif len(t.shape) == 3:
        return np.flip(t.permute(1,2,0).numpy(), axis=2)
    elif len(t.shape) == 4:
        return np.flip(t.permute(0,2,3,1).numpy(), axis=3)
def imwrite_tensor(t, name='test.png'):
    cv2.imwrite('./%s'%name, tensor2np(t.cpu()))
def imread_tensor(name='test'):
    return np2tensor(cv2.imread('./%s'%name))
def rot_hflip_img(img:torch.Tensor, rot_times:int=0, hflip:int=0):
    b=0 if len(img.shape)==3 else 1
    if hflip % 2 == 0:
        if rot_times % 4 == 0:    
            return img
        elif rot_times % 4 == 1:  
            return img.flip(b+1).transpose(b+1,b+2)
        elif rot_times % 4 == 2:  
            return img.flip(b+2).flip(b+1)
        else:               
            return img.flip(b+2).transpose(b+1,b+2)
    else:
        if rot_times % 4 == 0:    
            return img.flip(b+2)
        elif rot_times % 4 == 1:  
            return img.flip(b+1).flip(b+2).transpose(b+1,b+2)
        elif rot_times % 4 == 2:  
            return img.flip(b+1)
        else:               
            return img.transpose(b+1,b+2)
def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)
def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])
def psnr(img1, img2):
    if len(img1.shape) == 4:
        img1 = img1[0]
    if len(img2.shape) == 4:
        img2 = img2[0]
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    return peak_signal_noise_ratio(img1, img2, data_range=255)
def ssim(img1, img2):
    if len(img1.shape) == 4:
        img1 = img1[0]
    if len(img2.shape) == 4:
        img2 = img2[0]
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)
    img2 = np.clip(img2, 0, 255)
    img1 = np.clip(img1, 0, 255)
    return structural_similarity(img1, img2, multichannel=True, data_range=255)
def get_gaussian_2d_filter(window_size, sigma, channel=1, device=torch.device('cpu')):
    gauss = torch.ones(window_size, device=device)
    for x in range(window_size): gauss[x] = exp(-(x - window_size//2)**2/float(2*sigma**2))
    gauss = gauss.unsqueeze(1)
    filter2d = gauss.mm(gauss.t()).float()
    filter2d = (filter2d/filter2d.sum()).unsqueeze(0).unsqueeze(0)
    return filter2d.expand(channel, 1, window_size, window_size)
def get_mean_2d_filter(window_size, channel=1, device=torch.device('cpu')):
    window = torch.ones((window_size, window_size), device=device)
    window = (window/window.sum()).unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size)
def mean_conv2d(x, window_size=None, window=None, filter_type='gau', sigma=None, keep_sigma=False, padd=True):
    b_x = x.unsqueeze(0) if len(x.shape) == 3 else x
    if window is None:
        if sigma is None: sigma = (window_size-1)/6
        if filter_type == 'gau':
            window = get_gaussian_2d_filter(window_size, sigma=sigma, channel=b_x.shape[1], device=x.device)
        else:
            window = get_mean_2d_filter(window_size, channel=b_x.shape[1], device=x.device)
    else:
        window_size = window.shape[-1]
    if padd:
        pl = (window_size-1)//2
        b_x = F.pad(b_x, (pl,pl,pl,pl), 'reflect')
    m_b_x = F.conv2d(b_x, window, groups=b_x.shape[1])
    if keep_sigma:
        m_b_x /= (window**2).sum().sqrt()
    if len(x.shape) == 4:
        return m_b_x
    elif len(x.shape) == 3:
        return m_b_x.squeeze(0)