BLCNet: A Blind-spot Guided Local-Nonlocal Collaborative Self-supervised Denoising Network for SAR Images
=
Jiajie Ma, Lijun Zhao, Lianzhi Huo, Chunning Meng and Zhiqing Zhang 

Abstract—Synthetic Aperture Radar (SAR) images, due to their coherent imaging mechanism, are inevitably affected by severe speckle noise, which significantly limits subsequent target recognition and image interpretation. Existing self-supervised denoising methods struggle to balance the local continuity and non-local correlation characteristics of SAR images, thus constraining their denoising performance. To address this issue, this paper proposes a Blind-spot Guided Local–Nonlocal Collaborative Denoising Network (BLCNet), which achieves high-quality SAR image denoising without requiring clean images for supervision. The network employs a blind-spot masking mechanism to prevent direct reliance on target pixels. The Local Structure Modeling (LSM) branch extracts contextual information from neighboring regions via dilated convolutions, while the Nonlocal Relation Modeling (NRM) branch leverages a sparse Transformer to capture long-range yet semantically related dependencies. Furthermore, a Texture Enhancement Module (TEM) is introduced to reinforce fine structural details through bidirectional pooling and an attention mechanism. Extensive experimental results demonstrate that BLCNet outperforms current state-of-the-art self-supervised denoising methods on both real and synthetic SAR image datasets. 

Official Pytorch implementation of our model.

Train
--
`python tools/train.py --config`

Test
--
`python tools/test.py --config --ckpt`
