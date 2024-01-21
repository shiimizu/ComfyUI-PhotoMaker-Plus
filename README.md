# ComfyUI PhotoMaker 

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) reference implementation for [PhotoMaker](https://github.com/TencentARC/PhotoMaker).

PhotoMaker implementation that follows the ComfyUI way of doing things. The code is memory efficient, fast, and shouldn't break with Comfy updates.

<div align="center">

  <img width="1261" alt="example workflow" src="https://github.com/shiimizu/ComfyUI-PhotoMaker/assets/54494639/9ec3de2a-2fb3-4069-bff6-30b101568a78">
  
  <p>Example workflow that you can load in ComfyUI.</p>
</div>

<br>

# Installation
Download the model from [huggingface](https://huggingface.co/TencentARC/PhotoMaker) and place it in a `photomaker` folder in your `models` folder such as `ComfyUI/models/photomaker`.

1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
2. Clone this repo into `custom_nodes` by running the following commands in a terminal:
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/shiimizu/ComfyUI-PhotoMaker.git ComfyUI-PhotoMaker-SMZ
    ```

# Important news

**2024-01-21**
* Due to various node updates and changes, you may have to recreate the node.
* Removed `ref_images_path` input. Added a `PrepImagesForClipVisionFromPath` node.

**2024-01-18**
* No need to manually extract the LoRa that's inside the model anymore.

# Citation
```bibtex
@article{li2023photomaker,
  title={PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding},
  author={Li, Zhen and Cao, Mingdeng and Wang, Xintao and Qi, Zhongang and Cheng, Ming-Ming and Shan, Ying},
  booktitle={arXiv preprint arxiv:2312.04461},
  year={2023}
}
```
