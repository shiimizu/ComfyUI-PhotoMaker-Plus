import comfy.clip_vision
import comfy.clip_model
import comfy.model_management
from comfy.sd import CLIP
from comfy.clip_vision import ClipVisionModel
from itertools import zip_longest
from transformers import CLIPImageProcessor
from transformers.image_utils import PILImageResampling
import folder_paths
import torch
import os
from .utils import load_image, hook_all, tokenize_with_weights, prepImage, crop_image_pil, LoadImageCustom
from folder_paths import folder_names_and_paths, models_dir, supported_pt_extensions, add_model_folder_path
from torch import Tensor

folder_names_and_paths["photomaker"] = ([os.path.join(models_dir, "photomaker")], supported_pt_extensions)
add_model_folder_path("loras", folder_names_and_paths["photomaker"][0][0])

class PhotoMakerLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "name": (folder_paths.get_filename_list("photomaker"), ),
                             }}
    RETURN_TYPES = ("PHOTOMAKER",)
    FUNCTION = "load"

    CATEGORY = "photomaker"

    def load(self, name):
        hook_all()
        clip_path = folder_paths.get_full_path("photomaker", name)
        sd = torch.load(clip_path, map_location="cpu")
        if (id_encoder:=sd.get('id_encoder', None)):
            sd = id_encoder
        clip_vision = comfy.clip_vision.load_clipvision_from_sd(sd)
        hook_all(restore=True)
        return (clip_vision,)
    

class PhotoMakerEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "clip": ("CLIP",),
                    "photomaker": ("PHOTOMAKER",),
                    "image": ("IMAGE",),
                    "trigger_word": ("STRING", {"default": "img"}),
                    "text": ("STRING", {"multiline": True}),
                },
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "photomaker"

    @torch.no_grad()
    def encode(self, clip: CLIP, photomaker: ClipVisionModel, image: Tensor, trigger_word: str, text: str):
        if (num_id_images:=len(image)) == 0:
            raise ValueError("No image provided or found.")
        clip_vision = photomaker
        trigger_word=trigger_word.strip()
        tokens = clip.tokenize(text)
        class_tokens_mask = {}
        for key in tokens:
            clip_tokenizer = getattr(clip.tokenizer, f'clip_{key}', clip.tokenizer)
            tkwp = tokenize_with_weights(clip_tokenizer, text, return_tokens=True)
            # e.g.: 24157
            class_token = clip_tokenizer.tokenizer(trigger_word)["input_ids"][clip_tokenizer.tokens_start:-1][0]

            tmp=[]
            mask=[]
            num = num_id_images
            num_trigger_tokens_processed = 0
            for ls in tkwp:
                # recreate the list of pairs
                p = []
                pmask = []
                # remove consecutive duplicates
                newls = [ls[0]] + [curr for prev, curr in zip_longest(ls, ls[1:])
                                        if not (curr and prev and curr[0] == class_token and prev[0] == class_token)]
                if newls and newls[-1] is None: newls.pop()
                for pair in newls:
                    # Non-matches simply get appended to the list.
                    if pair[0] != class_token:
                        p.append(pair)
                        pmask.append(pair)
                    else:
                        # Found a match; append it to the previous list or main list's last list
                        num_trigger_tokens_processed += 1
                        if p:
                            # take the last element of the list we're creating and repeat it
                            pmask[-1] = (-1, pmask[-1][1])
                            if num-1 > 0:
                                p.extend([p[-1]] * (num-1))
                                pmask.extend([( -1, pmask[-1][1]  )] * (num-1))
                        else:
                            # The list we're cerating is empty so
                            # take the last element of the main list and then take its last element and repeat it
                            if tmp and tmp[-1]:
                                last_ls = tmp[-1]
                                last_pair = last_ls[-1]
                                mask[-1][-1] = (-1, mask[-1][-1][1])
                                if num-1 > 0:
                                    last_ls.extend([last_pair] * (num-1))
                                    mask[-1].extend([ (-1, mask[-1][-1][1]) ] * (num-1))
                if p: tmp.append(p)
                if pmask: mask.append(pmask)
            token_weight_pairs = tmp
            token_weight_pairs_mask = mask
            
            # send it back to be batched evenly
            token_weight_pairs = tokenize_with_weights(clip_tokenizer, text, _tokens=token_weight_pairs)
            token_weight_pairs_mask = tokenize_with_weights(clip_tokenizer, text, _tokens=token_weight_pairs_mask)
            tokens[key] = token_weight_pairs

            # Finalize the mask
            class_tokens_mask[key] = list(map(lambda a:  list(map(lambda b: b[0] < 0, a)), token_weight_pairs_mask))

        prompt_embeds, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        device_orig = prompt_embeds.device
        first_key = next(iter(class_tokens_mask.keys()))
        class_tokens_mask = torch.tensor(class_tokens_mask[first_key]).to(dtype=torch.bool, device=photomaker.load_device)
        
        if num_trigger_tokens_processed > 1:
            image = image.repeat([num_trigger_tokens_processed] + [1] * (len(image.shape) - 1))

        photomaker.model.prompt_embeds = prompt_embeds.to(photomaker.load_device)
        photomaker.model.class_tokens_mask = class_tokens_mask
        outputs = clip_vision.encode_image(image)
        cond = outputs.image_embeds.to(device=device_orig)

        return ([[cond, {"pooled_output": pooled}]],)


from .style_template import styles
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"

def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + ' ' + negative

class PhotoMakerStyles:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "style_name": (STYLE_NAMES, {"default": DEFAULT_STYLE_NAME}),
                },
                "optional": {
                    "positive": ("STRING", {"multiline": True, "forceInput": True}),
                    "negative": ("STRING", {"multiline": True, "forceInput": True}),
                },
            }
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("POSITIVE","NEGATIVE",)
    FUNCTION = "apply"

    CATEGORY = "photomaker"

    def apply(self, style_name, positive: str = '', negative: str = ''):
        positive, negative = apply_style(style_name, positive, negative)
        return (positive, negative)


class PrepImagesForClipVisionFromPath:
    def __init__(self) -> None:
        self.image_loader = LoadImageCustom()
        self.load_device = comfy.model_management.text_encoder_device()
        self.offload_device = comfy.model_management.text_encoder_offload_device()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "path": ("STRING", {"multiline": False}),
            "interpolation": (["nearest", "bilinear", "box", "bicubic", "lanczos", "hamming"], {"default": "lanczos"}),
            "crop_position": (["top", "bottom", "left", "right", "center", "pad"], {"default": "center"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prep_image"

    CATEGORY = "ipadapter"

    def prep_image(self, path:str, interpolation:str, crop_position,):
        image_path_list = []
        path = path.strip()
        if path:
            image_path_list = [path]
            if not (path.startswith("http://") or path.startswith("https://")) and os.path.isdir(path):
                image_basename_list = os.listdir(path)
                image_path_list = [
                    os.path.join(path, basename) 
                    for basename in image_basename_list
                    if not basename.startswith('.') and basename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
                ]
        if len(image_path_list) == 0:
            raise ValueError("No image provided or found.")

        interpolation=interpolation.upper()
        size = (224, 224)
        try:
            input_id_images = [img if (img:=load_image(image_path)).size == size else crop_image_pil(img, crop_position) for image_path in image_path_list]
            do_rescale = not all(img.size == size for img in input_id_images)
            resample = getattr(PILImageResampling, interpolation)
            clip_preprocess = CLIPImageProcessor(resample=resample, do_normalize=False, do_rescale=True, do_resize=True, do_center_crop=True)
            id_pixel_values = clip_preprocess(input_id_images, return_tensors="pt").pixel_values
            id_pixel_values = id_pixel_values.movedim(1,-1)
        except TypeError as err:
            print('[PhotoMaker]:', err)
            print('[PhotoMaker]: You may need to update transformers.')
            input_id_images = [self.image_loader.load_image(image_path)[0] for image_path in image_path_list]
            do_rescale = not all(img.shape[-3:-3+2] == size for img in input_id_images)
            if do_rescale:
                id_pixel_values = torch.cat([prepImage(img, interpolation=interpolation, crop_position=crop_position) for img in input_id_images])
            else:
                id_pixel_values = torch.cat(input_id_images)
        return (id_pixel_values,)

NODE_CLASS_MAPPINGS = {
    "PhotoMakerLoader": PhotoMakerLoader,
    "PhotoMakerEncode": PhotoMakerEncode,
    "PhotoMakerStyles": PhotoMakerStyles,
    "PrepImagesForClipVisionFromPath": PrepImagesForClipVisionFromPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoMakerLoader": "Load PhotoMaker",
    "PhotoMakerEncode": "PhotoMaker Encode",
    "PhotoMakerStyles": "Apply PhotoMaker Style",
    "PrepImagesForClipVisionFromPath": "Prepare Images For ClipVision From Path",
}