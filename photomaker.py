import comfy.clip_vision
import comfy.clip_model
import comfy.model_management
from comfy.sd import CLIP
from comfy.clip_vision import ClipVisionModel
import folder_paths
from .model import PhotoMakerIDEncoder
from copy import deepcopy
from .utils import load_image, hook_all, tokenize_with_weights
from itertools import zip_longest
from transformers import CLIPImageProcessor
from transformers.image_utils import PILImageResampling
import torch
import os
from folder_paths import folder_names_and_paths, models_dir, supported_pt_extensions, add_model_folder_path

folder_names_and_paths["photomaker"] = ([os.path.join(models_dir, "photomaker")], supported_pt_extensions)
add_model_folder_path("loras", folder_names_and_paths["photomaker"][0][0])

class PhotoMakerLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("photomaker"), ),
                             }}
    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "load_clip"

    CATEGORY = "photomaker"

    def load_clip(self, clip_name):
        hook_all()
        # comfy.clip_model.CLIPVisionModelProjection_original = comfy.clip_model.CLIPVisionModelProjection
        # comfy.clip_model.CLIPVisionModelProjection = PhotoMakerIDEncoder
        clip_path = folder_paths.get_full_path("photomaker", clip_name)
        # sd = comfy.clip_vision.load_torch_file(clip_path)
        sd = torch.load(clip_path, map_location="cpu")
        if 'id_encoder' in sd:
            sd = sd['id_encoder']
        # clip_vision =  comfy.clip_vision.load_clipvision_from_sd(sd, "id_encoder.", True)
        clip_vision = comfy.clip_vision.load_clipvision_from_sd(sd)
        # comfy.clip_model.CLIPVisionModelProjection = comfy.clip_model.CLIPVisionModelProjection_original
        hook_all(restore=True)
        return (clip_vision,)
    
    
class PhotoMakerEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "clip": ("CLIP",),
                    "clip_vision": ("CLIP_VISION",),
                    "text": ("STRING", {"multiline": True, "forceInput": True}),
                    "trigger_word": ("STRING", {"default": "img"}),
                    "ref_images_path": ("STRING", {"multiline": False, "placeholder": "optional"}),
                },
                "optional": {
                    "image": ("IMAGE",),
                }
        }
    RETURN_TYPES = ("CONDITIONING","CLIP_VISION_OUTPUT")
    FUNCTION = "encode"

    CATEGORY = "photomaker"

    def encode(self, clip: CLIP, clip_vision: ClipVisionModel, text: str, trigger_word: str, ref_images_path:str, image=None):

        input_id_images = image
        if ref_images_path != '':
            input_id_images=ref_images_path
            if not isinstance(input_id_images, list):
                input_id_images = [input_id_images]

            image_basename_list = os.listdir(ref_images_path)
            #image_path_list = sorted([os.path.join(ref_images_path, basename) for basename in image_basename_list])
            image_path_list = [
                os.path.join(ref_images_path, basename) 
                for basename in image_basename_list
                if not basename.startswith('.') and basename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))  # 只包括有效的图像文件
            ]

            input_id_images = [load_image(image_path) for image_path in image_path_list]
        
        if input_id_images is None or len(input_id_images) == 0:
            raise ValueError("No images provided or found.")
        num_id_images = len(input_id_images)
        # Resize to 224x224
        comfy.model_management.load_model_gpu(clip_vision.patcher)
        clip_vision.id_image_processor = CLIPImageProcessor(resample=PILImageResampling.LANCZOS, do_rescale=ref_images_path != '')
        id_pixel_values = clip_vision.id_image_processor(input_id_images, return_tensors="pt").pixel_values.float()
        id_pixel_values = id_pixel_values.to(device=clip_vision.load_device)

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

        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        prompt_embeds = cond.to(device=clip_vision.load_device)
        class_tokens_mask = torch.tensor(class_tokens_mask['l']).to(dtype=torch.bool, device=clip_vision.load_device)
        if (num_trigger_tokens_processed) > 1:
            id_pixel_values = id_pixel_values.repeat([num_trigger_tokens_processed] + [1] * (len(id_pixel_values.shape) - 1))

        # From clip_vision.encode_image
        out = clip_vision.model(id_pixel_values.unsqueeze(0), prompt_embeds, class_tokens_mask, intermediate_output=-2)

        outputs = comfy.clip_vision.Output()
        outputs["last_hidden_state"] = out[0].to(comfy.model_management.intermediate_device())
        outputs["penultimate_hidden_states"] = out[1].to(comfy.model_management.intermediate_device())
        outputs["image_embeds"] = out[2].to(comfy.model_management.intermediate_device())
        
        return ([[outputs.image_embeds, {"pooled_output": pooled}]], outputs)


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
    RETURN_NAMES = ("positive","negative",)
    FUNCTION = "apply"

    CATEGORY = "photomaker"

    def apply(self, style_name, positive: str = '', negative: str = ''):
        positive, negative = apply_style(style_name, positive, negative)
        return (positive, negative)

NODE_CLASS_MAPPINGS = {
    "PhotoMakerLoader": PhotoMakerLoader,
    "PhotoMakerEncode": PhotoMakerEncode,
    "PhotoMakerStyles": PhotoMakerStyles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoMakerLoader": "Load PhotoMaker",
    "PhotoMakerEncode": "PhotoMaker Encode",
    "PhotoMakerStyles": "Apply PhotoMaker style",
}