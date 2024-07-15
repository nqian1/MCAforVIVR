# Modality-Consistent Attention for Visible-infrared Vehicle Re-identification（2024 SPL）
# Network Architecture

![示例图片](https://github.com/nqian1/MCAforVIVR/blob/main/modal1.png)

# Requirements
<figure style="background-color: black; border-radius: 10px; padding: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
    <figcaption style="color: white; font-weight: bold;">Requirements</figcaption>
    <ul>
        <li> PyTorch 1.7, CUDA 11.1, and Python 3.8</li>
        <li>Ubuntu 18.04.5 LTS</li>
        <li> GeForce RTX 3090 GPU</li>
    </ul>
</figure>

# Training and Testing
```html
# Training Steps vivr trained on RGBN300 dataset
rgbn300/lctrain_senl.py

#Testing Steps vivr tested on RGBN300 dataset
rgbn300/test_senl.py



```
# Acknowledgment
```
Our project is based on the AGW baseline {https://github.com/mangye16/Cross-Modal-Re-ID-baseline} by Mang Ye, and we are grateful for it.
