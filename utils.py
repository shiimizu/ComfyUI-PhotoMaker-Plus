import os
import PIL
import PIL.Image
import PIL.ImageOps
import torch
from typing import Union
import torch.nn.functional as F
import torchvision.transforms as TT

def tokenize_with_trigger_word(tokens, weights, num_images, num_tokens, img_token, start_token=49406, end_token=49407, pad_token=0, max_len=77, return_mask=False):
    """
    Filters out the image token(s).
    Repeats the preceding token if any.
    Rebatches.
    """
    count = 0
    mask = (tokens != start_token) & (tokens != end_token) & (tokens != pad_token)
    clean_tokens, clean_tokens_mask = tokens[mask], weights[mask]
    img_token_indices = (clean_tokens == img_token).nonzero().view(-1)
    split = torch.tensor_split(clean_tokens, img_token_indices + 1, dim=-1)
    split_mask = torch.tensor_split(clean_tokens_mask, img_token_indices + 1, dim=-1)

    tt = []
    ww = []
    for chunk, chunk_mask in zip(split, split_mask):
        img_token_exists = chunk == img_token
        img_token_not_exists = ~img_token_exists
        pad_amount = img_token_exists.nonzero().view(-1).shape[0] * num_images * num_tokens
        chunk_clean, chunk_mask_clean = chunk[img_token_not_exists], chunk_mask[img_token_not_exists]
        if pad_amount > 0 and len(chunk_clean) > 0:
            count += 1
            tt.append(torch.nn.functional.pad(chunk_clean[:-1], (0, pad_amount), 'constant', chunk_clean[-1] if not return_mask else -1))
            ww.append(torch.nn.functional.pad(chunk_mask_clean[:-1], (0, pad_amount), 'constant', chunk_mask_clean[-1] if not return_mask else -1))
    
    if count == 0:
        return (tokens, weights, count)
    
    # rebatch and pad
    out = []
    outw = []
    one = torch.tensor([1.0])
    for tc, tcw in zip(torch.cat(tt).split(max_len - 2), torch.cat(ww).split(max_len - 2)):
        out.append(torch.cat([torch.tensor([start_token]), tc, torch.tensor([end_token])]))
        outw.append(torch.cat([one, tcw, one]))

    out = torch.nn.utils.rnn.pad_sequence(out, batch_first=True, padding_value=pad_token)
    outw = torch.nn.utils.rnn.pad_sequence(outw, batch_first=True, padding_value=1.0)

    out = torch.nn.functional.pad(out, (0, max(0, max_len - out.shape[1])), 'constant', pad_token)
    outw = torch.nn.functional.pad(outw, (0, max(0, max_len - outw.shape[1])), 'constant', 1.0)

    return (out, outw, count)

def load_pil_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            import requests
            img = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image_path = folder_paths.get_annotated_filepath(image)
            img = Image.open(image_path)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    return img

# from diffusers.utils import load_image
def load_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    image = load_pil_image(image)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

from PIL import Image, ImageSequence, ImageOps
import numpy as np
import folder_paths
from nodes import LoadImage
class LoadImageCustom(LoadImage):
    def load_image(self, image):
        # image_path = folder_paths.get_annotated_filepath(image)
        # img = Image.open(image_path)
        img = load_pil_image(image)
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

def crop_image_pil(image, crop_position):
    """
    Crop a PIL image based on the specified crop_position.

    Parameters:
    - image: PIL Image object
    - crop_position: One of "top", "bottom", "left", "right", "center", or "pad"

    Returns:
    - Cropped PIL Image object
    """

    width, height = image.size
    left, top, right, bottom = 0, 0, width, height

    if "pad" in crop_position:
        target_length = max(width, height)
        pad_l = max((target_length - width) // 2, 0)
        pad_t = max((target_length - height) // 2, 0)
        return ImageOps.expand(image, border=(pad_l, pad_t, target_length - width - pad_l, target_length - height - pad_t), fill=0)
    else:
        crop_size = min(width, height)
        x = (width - crop_size) // 2
        y = (height - crop_size) // 2

        if "top" in crop_position:
            bottom = top + crop_size
        elif "bottom" in crop_position:
            top = height - crop_size
            bottom = height
        elif "left" in crop_position:
            right = left + crop_size
        elif "right" in crop_position:
            left = width - crop_size
            right = width

        return image.crop((left, top, right, bottom))

def prepImages(images, *args, **kwargs):
    to_tensor = TT.ToTensor()
    images_ = []
    for img in images:
        image = to_tensor(img)
        if len(image.shape) <= 3: image.unsqueeze_(0)
        images_.append(prepImage(image.movedim(1,-1), *args, **kwargs))
    return torch.cat(images_)

def prepImage(image, interpolation="LANCZOS", crop_position="center", size=(224,224), sharpening=0.0, padding=0):
    _, oh, ow, _ = image.shape
    output = image.permute([0,3,1,2])

    if "pad" in crop_position:
        target_length = max(oh, ow)
        pad_l = (target_length - ow) // 2
        pad_r = (target_length - ow) - pad_l
        pad_t = (target_length - oh) // 2
        pad_b = (target_length - oh) - pad_t
        output = F.pad(output, (pad_l, pad_r, pad_t, pad_b), value=0, mode="constant")
    else:
        crop_size = min(oh, ow)
        x = (ow-crop_size) // 2
        y = (oh-crop_size) // 2
        if "top" in crop_position:
            y = 0
        elif "bottom" in crop_position:
            y = oh-crop_size
        elif "left" in crop_position:
            x = 0
        elif "right" in crop_position:
            x = ow-crop_size
        
        x2 = x+crop_size
        y2 = y+crop_size

        # crop
        output = output[:, :, y:y2, x:x2]

    # resize (apparently PIL resize is better than torchvision interpolate)
    imgs = []
    to_PIL_image = TT.ToPILImage()
    to_tensor = TT.ToTensor()
    for i in range(output.shape[0]):
        img = to_PIL_image(output[i])
        img = img.resize(size, resample=PIL.Image.Resampling[interpolation])
        imgs.append(to_tensor(img))
    output = torch.stack(imgs, dim=0)

    imgs = None # zelous GC
    
    if padding > 0:
        output = F.pad(output, (padding, padding, padding, padding), value=255, mode="constant")

    output = output.permute([0,2,3,1])

    return output
