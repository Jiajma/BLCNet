BLCNet: A Blind-spot Guided Local-Nonlocal Collaborative Self-supervised Denoising Network for SAR Images
=
Jiajie Ma, Lijun Zhao, Lianzhi Huo, Chunning Meng and Zhiqing Zhang 

Abstract—Synthetic Aperture Radar (SAR) images, due to their coherent imaging mechanism, are inevitably affected by severe speckle noise, which significantly limits subsequent target recognition and image interpretation. Existing self-supervised denoising methods struggle to balance the local continuity and non-local correlation characteristics of SAR images, thus constraining their denoising performance. To address this issue, this paper proposes a Blind-spot Guided Local–Nonlocal Collaborative Denoising Network (BLCNet), which achieves high-quality SAR image denoising without requiring clean images for supervision. The network employs a blind-spot masking mechanism to prevent direct reliance on target pixels. The Local Structure Modeling (LSM) branch extracts contextual information from neighboring regions via dilated convolutions, while the Nonlocal Relation Modeling (NRM) branch leverages a sparse Transformer to capture long-range yet semantically related dependencies. Furthermore, a Texture Enhancement Module (TEM) is introduced to reinforce fine structural details through bidirectional pooling and an attention mechanism. Extensive experimental results demonstrate that BLCNet outperforms current state-of-the-art self-supervised denoising methods on both real and synthetic SAR image datasets. 

Official Pytorch implementation of our model.

[图片15.tif](https://github.com/user-attachments/files/24843226/15.tif)


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

## Full-Reference Evaluation Metrics

We use **Peak Signal-to-Noise Ratio (PSNR)** and the **Structural Similarity Index (SSIM)** as full-reference image quality metrics to evaluate the despeckling performance on **synthetic SAR images**.

### Peak Signal-to-Noise Ratio (PSNR)

PSNR measures the pixel-wise fidelity between the denoised image and the reference image. A higher PSNR indicates a smaller reconstruction error and thus better denoising performance. Let $\hat{I}$ denote the denoised image, $I$ the reference image, and $\mathrm{MSE}$ the mean squared error. $\mathrm{MAX}$ is the maximum possible pixel value:

$$
\mathrm{MSE}=\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}\left[I(i,j)-\hat{I}(i,j)\right]^2 \qquad 
$$

$$
\mathrm{PSNR}=20\cdot\log_{10}\left(\frac{\mathrm{MAX}}{\sqrt{\mathrm{MSE}}}\right) \qquad 
$$

### Structural Similarity Index (SSIM)

SSIM evaluates the similarity between two images in terms of **luminance**, **contrast**, and **structure**, and is more consistent with human visual perception than purely pixel-wise metrics. A higher SSIM indicates that the denoised image is closer to the reference image. Let $\mu_x$ and $\mu_y$ be the mean intensities of images $x$ and $y$, $\sigma_x$ and $\sigma_y$ their standard deviations, and $\sigma_{xy}$ the covariance. Constants $C_1$ and $C_2$ are included to stabilize the computation:

$$
\mathrm{SSIM}(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)} \qquad 
$$

## Ratio-Image-Based Speckle Statistics

We characterize the distribution of the speckle component estimated from ratio images and compare it with the assumed speckle distribution under the multiplicative noise model.

We assume the observed SAR image follows

$$
Y = X \cdot N,
$$

where $X$ is the ideal noise-free (clean) image and $N$ denotes speckle noise (typically normalized to have a **mean value of 1**).

### Ratio Image Construction

To describe the speckle component implied by a despeckling result, we construct a **ratio image**:

$$
R = \frac{Y}{\hat{X}+\varepsilon},
$$

where $\hat{X}$ is the despeckled output and $\varepsilon$ is a small constant introduced to avoid division by zero. If $\hat{X}\approx X$, then

$$
R \approx \frac{X\cdot N}{X}=N,
$$

meaning that $R$ can be regarded as an estimate of the speckle component $N$, and its histogram should match the distribution of “pure speckle”.

### Pixel-Statistic and Histogram-Statistic Comparison

In our implementation, we report:
- The **mean** and **standard deviation** computed directly from the **pixel values of the ratio image** $R$ (after normalization);
- The **Bhattacharyya distance** computed between the **histogram of $R$** and the **histogram of a reference “pure speckle” image** (or noise factor) $N$ synthesized under the assumed noise model.

Specifically, let the normalized histograms (probability mass functions) be $p(i)$ and $q(i)$. The Bhattacharyya coefficient is defined as

$$
BC(p,q)=\sum_i \sqrt{p(i)\,q(i)},
$$

and the corresponding Bhattacharyya distance is

$$
D_B(p,q)=-\ln\big(BC(p,q)\big).
$$

A smaller $D_B$ indicates that the histogram of the ratio image is closer to the histogram of “pure speckle”, i.e., the noise component separated (removed) by the method is statistically closer to pure speckle.
