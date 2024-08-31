import comfy.clip_vision
import comfy.clip_model
import comfy.model_management
import comfy.utils
from comfy.sd import CLIP
from transformers import CLIPImageProcessor
from transformers.image_utils import PILImageResampling
from collections import Counter
import folder_paths
import torch
import os
from .model import PhotoMakerIDEncoder
from .model_v2 import PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken
from .utils import LoadImageCustom, load_image, prepImage, crop_image_pil, tokenize_with_trigger_word
from folder_paths import folder_names_and_paths, models_dir, supported_pt_extensions, add_model_folder_path
from torch import Tensor
import hashlib
from typing import Union
from .insightface_package import analyze_faces, insightface_loader
import numpy as np
import torchvision.transforms.v2 as T

INSIGHTFACE_DIR = os.path.join(models_dir, "insightface")

folder_names_and_paths["photomaker"] = ([os.path.join(models_dir, "photomaker")], supported_pt_extensions)
add_model_folder_path("loras", folder_names_and_paths["photomaker"][0][0])

class PhotoMakerLoaderPlus:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "photomaker_model_name": (folder_paths.get_filename_list("photomaker"), ),
                             }}
    RETURN_TYPES = ("PHOTOMAKER",)
    FUNCTION = "load_photomaker_model"

    CATEGORY = "PhotoMaker"

    def load_photomaker_model(self, photomaker_model_name):
        photomaker_model_path = folder_paths.get_full_path("photomaker", photomaker_model_name)
        if 'v1' in photomaker_model_name:
            photomaker_model = PhotoMakerIDEncoder()
        else:
            photomaker_model = PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken()
        data = comfy.utils.load_torch_file(photomaker_model_path, safe_load=True)
        if "id_encoder" in data:
            data = data["id_encoder"]
        photomaker_model.load_state_dict(data)
        return (photomaker_model,)

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
        tokens = clip.tokenize(text)
        class_tokens_mask = {}
        out_tokens = {}
        for key, val in tokens.items():
            clip_tokenizer = getattr(clip.tokenizer, f'clip_{key}', clip.tokenizer)
            img_token = clip_tokenizer.tokenizer(trigger_word.strip(), truncation=False, add_special_tokens=False)["input_ids"][0] # only get the first token
            _tokens = torch.tensor([[tpy[0] for tpy in tpy_]  for tpy_ in val ] , dtype=torch.int32)
            _weights = torch.tensor([[tpy[1] for tpy in tpy_]  for tpy_ in val] , dtype=torch.float32)
            start_token = clip_tokenizer.start_token
            end_token = clip_tokenizer.end_token
            pad_token = clip_tokenizer.pad_token
            
            tokens_mask = tokenize_with_trigger_word(_tokens, _weights, num_images,img_token,start_token, end_token, pad_token, return_mask=True)[0]
            tokens_new, weights_new, num_trigger_tokens_processed = tokenize_with_trigger_word(_tokens, _weights, num_images,img_token,start_token, end_token, pad_token)
            token_weight_pairs = [[(tt,ww) for tt,ww in zip(x.tolist(), y.tolist())] for x,y in zip(tokens_new, weights_new)]
            mask = (tokens_mask == -1).tolist()
            class_tokens_mask[key] = mask
            out_tokens[key] = token_weight_pairs

        cond, pooled = clip.encode_from_tokens(out_tokens, return_pooled=True)
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
            cond = photomaker(id_pixel_values=pixel_values.unsqueeze(0), prompt_embeds=cond.to(photomaker.load_device),
                        class_tokens_mask=torch.tensor(class_tokens_mask, dtype=torch.bool, device=photomaker.load_device))
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


from .style_template import styles

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
            print('[PhotoMaker]:', err)
            print('[PhotoMaker]: You may need to update transformers.')
            input_id_images = [self.image_loader.load_image(image_path)[0] for image_path in image_path_list]
            do_resize = not all(img.shape[-3:-3+2] == size for img in input_id_images)
            if do_resize:
                id_pixel_values = torch.cat([prepImage(img, interpolation=interpolation, crop_position=crop_position) for img in input_id_images])
            else:
                id_pixel_values = torch.cat(input_id_images)
        return (id_pixel_values,)

# supported = False
# try:
#     from comfy_extras.nodes_photomaker import PhotoMakerLoader as _PhotoMakerLoader
#     supported = True
# except Exception: ...

NODE_CLASS_MAPPINGS = {
#    **({} if supported else {"PhotoMakerLoader": PhotoMakerLoaderPlus}),
    "PhotoMakerLoaderPlus": PhotoMakerLoaderPlus,
    "PhotoMakerEncodePlus": PhotoMakerEncodePlus,
    "PhotoMakerStyles": PhotoMakerStyles,
    "PrepImagesForClipVisionFromPath": PrepImagesForClipVisionFromPath,
    "PhotoMakerInsightFaceLoader": PhotoMakerInsightFaceLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
#    **({} if supported else {"PhotoMakerLoader": "Load PhotoMaker"}),
    "PhotoMakerLoaderPlus": "PhotoMaker Loader Plus",
    "PhotoMakerEncodePlus": "PhotoMaker Encode Plus",
    "PhotoMakerStyles": "Apply PhotoMaker Style",
    "PrepImagesForClipVisionFromPath": "Prepare Images For CLIP Vision From Path",
    "PhotoMakerInsightFaceLoader": "PhotoMaker InsightFace Loader",
}
