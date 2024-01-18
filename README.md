# ComfyUI-PhotoMaker 

Unofficial implementation of [PhotoMaker](https://github.com/TencentARC/PhotoMaker) for ComfyUI.

<img width="1261" alt="example" src="https://github.com/shiimizu/ComfyUI-PhotoMaker/assets/54494639/0a164254-da1c-4aa6-9e6b-957a6118a09a">

---

Download the [model](https://huggingface.co/TencentARC/PhotoMaker) and place it in your `models` folder such as `ComfyUI/models/photomaker`.

Extract the LoRa and place it in your `loras` folder:
```python
import torch
from safetensors.torch import save_file
sd = torch.load("/path/to/photomaker-v1.bin", map_location="cpu")
save_file(sd["lora_weights"], "ComfyUI/models/loras/photomaker-v1-lora.safetensors")
```

# Citation
```bibtex
@article{li2023photomaker,
  title={PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding},
  author={Li, Zhen and Cao, Mingdeng and Wang, Xintao and Qi, Zhongang and Cheng, Ming-Ming and Shan, Ying},
  booktitle={arXiv preprint arxiv:2312.04461},
  year={2023}
}
```
