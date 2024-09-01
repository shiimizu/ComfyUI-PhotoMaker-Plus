# ComfyUI PhotoMaker Plus

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) implementation for [PhotoMaker](https://github.com/TencentARC/PhotoMaker).

PhotoMaker implementation that follows the ComfyUI way of doing things. The code is memory efficient, fast, and shouldn't break with Comfy updates.

<div align="center">

  <img width="1261" alt="example workflow" src="https://github.com/user-attachments/assets/0b05d24f-ec3a-4be7-9e93-9169c66c65f5">
  
</div>

## Installation

1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
2. Install [onnxruntime](https://onnxruntime.ai/docs/install/#python-installs) and `insightface`.
3. Install through [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) or clone this repo into `custom_nodes` by running the following commands in a terminal:
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/shiimizu/ComfyUI-PhotoMaker-Plus.git
    ```
4. Download the model(s) from Hugging Face ([V1](https://huggingface.co/TencentARC/PhotoMaker), [V2](https://huggingface.co/TencentARC/PhotoMaker-V2)) and place it in a `photomaker` folder in your `models` folder such as `ComfyUI/models/photomaker`.
5. Check out the [example workflows](https://github.com/shiimizu/ComfyUI-PhotoMaker-Plus/tree/main/examples).

## Features of this `Plus` version

* Better face resemblance by using `CLIPImageProcessor` like in the original code.
* Customizable trigger word
* Allows multiple trigger words in the prompt
* Extra nodes such as `PhotoMakerStyles` and `PrepImagesForClipVisionFromPath`

## Important news

**2024-09-01**
* A `PhotoMakerLoraLoaderPlus` node was added. Use that to load the LoRA.

**2024-07-26**
* Support for PhotoMaker V2. This uses InsightFace, so make sure to use the new `PhotoMakerLoaderPlus` and `PhotoMakerInsightFaceLoader` nodes.

**2024-01-24**
* [Official support](https://github.com/comfyanonymous/ComfyUI/commit/d1533d9c0f1dde192f738ef1b745b15f49f41e02) for PhotoMaker landed in ComfyUI. Therefore, this repo's name has been changed. The `PhotoMakerEncode` node is also now `PhotoMakerEncodePlus`.

**2024-01-21**
* Due to various node updates and changes, you may have to recreate the node.
* Removed `ref_images_path` input. Added a `PrepImagesForClipVisionFromPath` node.

**2024-01-18**
* No need to manually extract the LoRA that's inside the model anymore.

## Citation
```bibtex
@article{li2023photomaker,
  title={PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding},
  author={Li, Zhen and Cao, Mingdeng and Wang, Xintao and Qi, Zhongang and Cheng, Ming-Ming and Shan, Ying},
  booktitle={arXiv preprint arxiv:2312.04461},
  year={2023}
}
```
