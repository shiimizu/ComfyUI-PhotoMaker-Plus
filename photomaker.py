import torch
import hashlib
import os
import logging
import numpy as np
import comfy.clip_vision
import comfy.clip_model
import comfy.model_management
import comfy.utils
import comfy.sd
import folder_paths
import torchvision.transforms.v2 as T
from comfy.sd import CLIP
from typing import Union
from collections import Counter
from torch import Tensor
from transformers import CLIPImageProcessor
from transformers.image_utils import PILImageResampling
from .insightface_package import analyze_faces, insightface_loader
from .model import PhotoMakerIDEncoder
from .model_v2 import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken
from .utils import LoadImageCustom, load_image, prepImage, crop_image_pil, tokenize_with_trigger_word
from .style_template import styles

class PhotoMakerLoaderPlus:
    def __init__(self):
        self.loaded_lora = None
        self.loaded_clipvision = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "photomaker_model_name": (folder_paths.get_filename_list("photomaker"), ),
            },
        }
    RETURN_TYPES = ("PHOTOMAKER", )
    FUNCTION = "load_photomaker_model"

    CATEGORY = "PhotoMaker"

    def load_photomaker_model(self, photomaker_model_name):
        self.load_data(None, None, photomaker_model_name, 0, 0)[0]
        if 'qformer_perceiver.token_norm.weight' in self.loaded_clipvision[1].keys():
            photomaker_model = PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken()
        else:
            photomaker_model = PhotoMakerIDEncoder()
        photomaker_model.load_state_dict(self.loaded_clipvision[1])
        photomaker_model.loader = self
        photomaker_model.filename = photomaker_model_name
        return (photomaker_model,)

    def load_data(self, model, clip, name, strength_model, strength_clip):
        model_lora, clip_lora = model, clip

        path = folder_paths.get_full_path("photomaker", name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp
                temp = self.loaded_clipvision
                self.loaded_clipvision = None
                del temp

        if lora is None:
            data = comfy.utils.load_torch_file(path, safe_load=True)
            clipvision = data.get("id_encoder", None)
            lora = data.get("lora_weights", None)
            self.loaded_lora = (path, lora)
            self.loaded_clipvision = (path, clipvision)

        if model is not None and (strength_model > 0 or strength_clip > 0):
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

class PhotoMakerLoraLoaderPlus:
    def __init__(self):
        self.loaded_lora = None
        self.loaded_clipvision = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "model": ("MODEL",),
                "photomaker": ("PHOTOMAKER",),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
        }
    RETURN_TYPES = ("MODEL", )
    FUNCTION = "load_photomaker_lora"

    CATEGORY = "PhotoMaker"

    def load_photomaker_lora(self, model, photomaker, lora_strength):
        return (photomaker.loader.load_data(model, None, photomaker.filename, lora_strength, 0)[0],)

class PhotoMakerInsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"], ),
            },
        }

    RETURN_TYPES = ("INSIGHTFACE",)
    FUNCTION = "load_insightface"
    CATEGORY = "PhotoMaker"

    def load_insightface(self, provider):
        return (insightface_loader(provider),)

class PhotoMakerEncodePlus:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "clip": ("CLIP",),
                    "photomaker": ("PHOTOMAKER",),
                    "image": ("IMAGE",),
                    "trigger_word": ("STRING", {"default": "img"}),
                    "text": ("STRING", {"multiline": True, "default": "photograph of a man img", "dynamicPrompts": True}),
                },
                "optional": {
                    "insightface_opt": ("INSIGHTFACE",),
                },
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_photomaker"

    CATEGORY = "PhotoMaker"

    @torch.no_grad()
    def apply_photomaker(self, clip: CLIP, photomaker: Union[PhotoMakerIDEncoder, PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken], image: Tensor, trigger_word: str, text: str, insightface_opt=None):
        if (num_images := len(image)) == 0:
            raise ValueError("No image provided or found.")
        trigger_word=trigger_word.strip()
        tokens = clip.tokenize(text)
        class_tokens_mask = {}
        out_tokens = {}
        num_tokens = getattr(photomaker, 'num_tokens', 1)
        for key, val in tokens.items():
            clip_tokenizer = getattr(clip.tokenizer, f'clip_{key}', clip.tokenizer)
            img_token = clip_tokenizer.tokenizer(trigger_word, truncation=False, add_special_tokens=False)["input_ids"][0] # only get the first token
            _tokens = torch.tensor([[tpy[0] for tpy in tpy_]  for tpy_ in val ] , dtype=torch.int32)
            _weights = torch.tensor([[tpy[1] for tpy in tpy_]  for tpy_ in val] , dtype=torch.float32)
            start_token = clip_tokenizer.start_token
            end_token = clip_tokenizer.end_token
            pad_token = clip_tokenizer.pad_token
            
            tokens_mask = tokenize_with_trigger_word(_tokens, _weights, num_images, num_tokens, img_token,start_token, end_token, pad_token, return_mask=True)[0]
            tokens_new, weights_new, num_trigger_tokens_processed = tokenize_with_trigger_word(_tokens, _weights, num_images, num_tokens, img_token,start_token, end_token, pad_token)
            token_weight_pairs = [[(tt,ww) for tt,ww in zip(x.tolist(), y.tolist())] for x,y in zip(tokens_new, weights_new)]
            mask = (tokens_mask == -1).tolist()
            class_tokens_mask[key] = mask
            out_tokens[key] = token_weight_pairs

        cond, pooled = clip.encode_from_tokens(out_tokens, return_pooled=True)
        if num_trigger_tokens_processed == 0 or not trigger_word:
            logging.warning("\033[33mWarning:\033[0m No trigger token found.")
            return ([[cond, {"pooled_output": pooled}]],)

        prompt_embeds = cond
        device_orig = prompt_embeds.device
        first_key = next(iter(tokens.keys()))
        class_tokens_mask = class_tokens_mask[first_key]
        if num_trigger_tokens_processed > 1:
            image = image.repeat([num_trigger_tokens_processed] + [1] * (len(image.shape) - 1))

        photomaker = photomaker.to(device=photomaker.load_device)

        image.clamp_(0.0, 1.0)
        input_id_images = image
        _, h, w, _ = image.shape
        do_resize = (h, w) != (224, 224)
        image_bak = image
        try:
            if do_resize:
                clip_preprocess = CLIPImageProcessor(resample=PILImageResampling.LANCZOS, do_normalize=False, do_rescale=False, do_convert_rgb=False)
                image = clip_preprocess(image, return_tensors="pt").pixel_values.movedim(1,-1)
        except RuntimeError as e:
            image = image_bak
        pixel_values = comfy.clip_vision.clip_preprocess(image.to(photomaker.load_device)).float()

        if photomaker.__class__.__name__ == 'PhotoMakerIDEncoder':
            cond = photomaker(id_pixel_values=pixel_values.unsqueeze(0),
                prompt_embeds=cond.to(photomaker.load_device),
                class_tokens_mask=torch.tensor(class_tokens_mask, dtype=torch.bool, device=photomaker.load_device).unsqueeze(0))
        else:
            if insightface_opt is None:
                raise ValueError(f"InsightFace is required for PhotoMaker V2")
            face_detector = insightface_opt
            if not hasattr(face_detector, 'get_'):
                face_detector.get_ = face_detector.get
                def get(self, img, max_num=0, det_size=(640, 640)):
                    if det_size is not None:
                        self.det_model.input_size = det_size
                    return self.get_(img, max_num)
                face_detector.get = get.__get__(face_detector, face_detector.__class__)

            id_embed_list = []

            ToPILImage = T.ToPILImage()
            def tensor_to_pil_np(_img):
                nonlocal ToPILImage
                img_pil = ToPILImage(_img.movedim(-1,0))
                if img_pil.mode != 'RGB': img_pil = img_pil.convert('RGB')
                return np.asarray(img_pil)

            for img in input_id_images:
                faces = analyze_faces(face_detector, tensor_to_pil_np(img))
                if len(faces) > 0:
                    id_embed_list.append(torch.from_numpy((faces[0]['embedding'])))

            if len(id_embed_list) == 0:
                raise ValueError(f"No face detected in input image pool")

            id_embeds = torch.stack(id_embed_list).to(device=photomaker.load_device)

            class_tokens_mask=torch.tensor(class_tokens_mask, dtype=torch.bool, device=photomaker.load_device).unsqueeze(0)
            cond = photomaker(id_pixel_values=pixel_values.unsqueeze(0),
                            prompt_embeds=cond.to(photomaker.load_device),
                            class_tokens_mask=class_tokens_mask,
                            id_embeds=id_embeds)
        cond = cond.to(device=device_orig)

        return ([[cond, {"pooled_output": pooled}]],)

class PhotoMakerStyles:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "style_name": (list(styles.keys()), {"default": "Photographic (Default)"}),
                },
                "optional": {
                    "positive": ("STRING", {"multiline": True, "forceInput": True, "dynamicPrompts": True}),
                    "negative": ("STRING", {"multiline": True, "forceInput": True, "dynamicPrompts": True}),
                },
            }
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("POSITIVE","NEGATIVE",)
    FUNCTION = "apply_photomaker_style"

    CATEGORY = "PhotoMaker"

    def apply_photomaker_style(self, style_name, positive: str = '', negative: str = ''):
        p, n = styles.get(style_name, "Photographic (Default)")
        return p.replace("{prompt}", positive), n + ' ' + negative

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

    @classmethod
    def IS_CHANGED(s, path:str, interpolation, crop_position):
        image_path_list = s.get_images_paths(path)
        hashes = []
        for image_path in image_path_list:
            if not (path.startswith("http://") or path.startswith("https://")):
                m = hashlib.sha256()
                with open(image_path, 'rb') as f:
                    m.update(f.read())
                hashes.append(m.digest().hex())
        return Counter(hashes)

    @classmethod
    def VALIDATE_INPUTS(s, path:str, interpolation, crop_position):
        image_path_list = s.get_images_paths(path)
        if len(image_path_list) == 0:
            return "No image provided or found."
        return True

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prep_images_for_clip_vision_from_path"

    CATEGORY = "image"

    @classmethod
    def get_images_paths(self, path:str):
        image_path_list = []
        path = path.strip()
        if path:
            image_path_list = [path]
            if not (path.startswith("http://") or path.startswith("https://")) and os.path.isdir(path):
                image_basename_list = os.listdir(path)
                image_path_list = [
                    os.path.join(path, basename) 
                    for basename in image_basename_list
                    if not basename.startswith('.') and basename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif'))
                ]
        return image_path_list

    def prep_images_for_clip_vision_from_path(self, path:str, interpolation:str, crop_position,):
        image_path_list = self.get_images_paths(path)
        if len(image_path_list) == 0:
            raise ValueError("No image provided or found.")

        interpolation=interpolation.upper()
        size = (224, 224)
        try:
            input_id_images = [img if (img:=load_image(image_path)).size == size else crop_image_pil(img, crop_position) for image_path in image_path_list]
            do_resize = not all(img.size == size for img in input_id_images)
            resample = getattr(PILImageResampling, interpolation)
            clip_preprocess = CLIPImageProcessor(resample=resample, do_normalize=False, do_resize=do_resize)
            id_pixel_values = clip_preprocess(input_id_images, return_tensors="pt").pixel_values.movedim(1,-1)
        except TypeError as err:
            logging.warning('[PhotoMaker]:', err)
            logging.warning('[PhotoMaker]: You may need to update transformers.')
            input_id_images = [self.image_loader.load_image(image_path)[0] for image_path in image_path_list]
            do_resize = not all(img.shape[-3:-3+2] == size for img in input_id_images)
            if do_resize:
                id_pixel_values = torch.cat([prepImage(img, interpolation=interpolation, crop_position=crop_position) for img in input_id_images])
            else:
                id_pixel_values = torch.cat(input_id_images)
        return (id_pixel_values,)


NODE_CLASS_MAPPINGS = {
    "PhotoMakerLoaderPlus": PhotoMakerLoaderPlus,
    "PhotoMakerEncodePlus": PhotoMakerEncodePlus,
    "PhotoMakerStyles": PhotoMakerStyles,
    "PhotoMakerLoraLoaderPlus": PhotoMakerLoraLoaderPlus,
    "PrepImagesForClipVisionFromPath": PrepImagesForClipVisionFromPath,
    "PhotoMakerInsightFaceLoader": PhotoMakerInsightFaceLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoMakerLoaderPlus": "PhotoMaker Loader Plus",
    "PhotoMakerEncodePlus": "PhotoMaker Encode Plus",
    "PhotoMakerStyles": "Apply PhotoMaker Style",
    "PhotoMakerLoraLoaderPlus": "PhotoMaker LoRA Loader Plus",
    "PrepImagesForClipVisionFromPath": "Prepare Images For CLIP Vision From Path",
    "PhotoMakerInsightFaceLoader": "PhotoMaker InsightFace Loader",
}