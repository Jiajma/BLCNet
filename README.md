BLCNet: A Blind-spot Guided Local-Nonlocal Collaborative Self-supervised Denoising Network for SAR Images
=
Jiajie Ma, Lijun Zhao, Lianzhi Huo, Chunning Meng and Zhiqing Zhang 

Abstract—Synthetic Aperture Radar (SAR) images, due to their coherent imaging mechanism, are inevitably affected by severe speckle noise, which significantly limits subsequent target recognition and image interpretation. Existing self-supervised denoising methods struggle to balance the local continuity and non-local correlation characteristics of SAR images, thus constraining their denoising performance. To address this issue, this paper proposes a Blind-spot Guided Local–Nonlocal Collaborative Denoising Network (BLCNet), which achieves high-quality SAR image denoising without requiring clean images for supervision. The network employs a blind-spot masking mechanism to prevent direct reliance on target pixels. The Local Structure Modeling (LSM) branch extracts contextual information from neighboring regions via dilated convolutions, while the Nonlocal Relation Modeling (NRM) branch leverages a sparse Transformer to capture long-range yet semantically related dependencies. Furthermore, a Texture Enhancement Module (TEM) is introduced to reinforce fine structural details through bidirectional pooling and an attention mechanism. Extensive experimental results demonstrate that BLCNet outperforms current state-of-the-art self-supervised denoising methods on both real and synthetic SAR image datasets. 

Official Pytorch implementation of our model.

Train
--
`python tools/train.py --train_dir /dir --val_dir /dir --save_model_path /dir`

Test
--
`python tools/test.py --test_dir /dir --save_dir /dir --model_path /dir`

## No-Reference Evaluation Metrics

We adopt three no-reference image quality metrics—**Equivalent Number of Looks (ENL)**, **Mean of Ratio (MOR)**, and **Contrast-to-Noise Ratio (CNR)**—to quantitatively evaluate the despeckling performance on **real SAR images**.

### Equivalent Number of Looks (ENL)

ENL is commonly used to characterize the speckle level in **homogeneous** regions. A larger ENL indicates a smoother image and less speckle. Let $\mu$ and $\sigma^2$ denote the mean and variance of the pixel intensities within a homogeneous region, respectively:

$$
\mathrm{ENL} = \frac{\mu^2}{\sigma^2} \qquad 
$$

### Mean of Ratio (MOR)

MOR measures whether the denoised image preserves the **overall brightness** of the original noisy image. An ideal despeckling result should keep $\mathrm{MOR}$ close to 1:

$$
\mathrm{MOR} = \frac{\mu_{\mathrm{denoised}}}{\mu_{\mathrm{noisy}}} \qquad 
$$

### Contrast-to-Noise Ratio (CNR)

CNR quantifies the contrast between a **target** region and a **background** region relative to the noise level of the background. Let $\mu_{\mathrm{target}}$ and $\mu_{\mathrm{background}}$ be the mean intensities of the target and background regions, respectively, and let $\sigma_{\mathrm{background}}$ be the standard deviation of the background region. A higher CNR indicates better separability between target and background after denoising:

$$
\mathrm{CNR} = 20 \cdot \log_{10}\left(\frac{\left|\mu_{\mathrm{target}}-\mu_{\mathrm{background}}\right|}{\sigma_{\mathrm{background}}}\right) \qquad 
$$

