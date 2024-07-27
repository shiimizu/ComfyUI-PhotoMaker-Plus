import os
import sys
import PIL
import PIL.Image
import PIL.ImageOps
import inspect
import importlib
import types
import functools
from textwrap import dedent, indent
from copy import copy
import torch
from typing import List, Union
from collections import namedtuple
from .model import PhotoMakerIDEncoder
import comfy.sd1_clip
from comfy.sd1_clip import escape_important, token_weights, unescape_important
import torch.nn.functional as F
import torchvision.transforms as TT

Hook = namedtuple('Hook', ['fn', 'module_name', 'target', 'orig_key', 'module_name_nt', 'module_name_unix'])

def hook_clip_model_CLIPVisionModelProjection():
    return create_hook(PhotoMakerIDEncoder, 'comfy.clip_model', 'CLIPVisionModelProjection')

def hook_tokenize_with_weights():
    import comfy.sd1_clip
    if not hasattr(comfy.sd1_clip.SDTokenizer, 'tokenize_with_weights_original'):
        comfy.sd1_clip.SDTokenizer.tokenize_with_weights_original = comfy.sd1_clip.SDTokenizer.tokenize_with_weights
    comfy.sd1_clip.SDTokenizer.tokenize_with_weights = tokenize_with_weights
    return create_hook(tokenize_with_weights, 'comfy.sd1_clip', 'SDTokenizer.tokenize_with_weights')

def hook_load_torch_file():
    import comfy.utils
    if not hasattr(comfy.utils, 'load_torch_file_original'):
        comfy.utils.load_torch_file_original = comfy.utils.load_torch_file
    replace_str="""
    if sd.get('id_encoder', None) and (lora_weights:=sd.get('lora_weights', None)) and len(sd) == 2:
        def find_outer_instance(target:str, target_type):
            import inspect
            frame = inspect.currentframe()
            i = 0
            while frame and i < 5:
                if (found:=frame.f_locals.get(target, None)) is not None:
                    if isinstance(found, target_type):
                        return found
                frame = frame.f_back
                i += 1
            return None
        if find_outer_instance('lora_name', str) is not None:
            sd = lora_weights
    return sd"""
    source = inspect.getsource(comfy.utils.load_torch_file_original)
    modified_source = source.replace("return sd", replace_str)
    fn = write_to_file_and_return_fn(comfy.utils.load_torch_file_original, modified_source, 'w')
    return create_hook(fn, 'comfy.utils')

def create_hook(fn, module_name, target = None, orig_key = None):
    if target is None: target = fn.__name__
    if orig_key is None: orig_key = f'{target}_original'
    module_name_nt = '\\'.join(module_name.split('.'))
    module_name_unix = '/'.join(module_name.split('.'))
    return Hook(fn, module_name, target, orig_key, module_name_nt, module_name_unix)

def hook_all(restore=False, hooks = None):
    if hooks is None:
        hooks: List[Hook] = [
            hook_clip_model_CLIPVisionModelProjection(),
        ]
    for m in list(sys.modules.keys()):
        for hook in hooks:
            if hook is None:
                continue
            if hook.module_name == m or (os.name != 'nt' and m.endswith(hook.module_name_unix)) or (os.name == 'nt' and m.endswith(hook.module_name_nt)):
                if hasattr(sys.modules[m], hook.target):
                    if not hasattr(sys.modules[m], hook.orig_key):
                        if (orig_fn:=getattr(sys.modules[m], hook.target, None)) is not None:
                            setattr(sys.modules[m], hook.orig_key, orig_fn)
                    if restore:
                        setattr(sys.modules[m], hook.target, getattr(sys.modules[m], hook.orig_key, None))
                    else:
                        setattr(sys.modules[m], hook.target, hook.fn)

def tokenize_with_weights(self: comfy.sd1_clip.SDTokenizer, text:str, return_word_ids=False, tokens=None, return_tokens=False):
    '''
    Takes a prompt and converts it to a list of (token, weight, word id) elements.
    Tokens can both be integer tokens and pre computed CLIP tensors.
    Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
    Returned list has the dimensions NxM where M is the input size of CLIP
    '''
    if self.pad_with_end:
        pad_token = self.end_token
    else:
        pad_token = 0

    if tokens is None:
        tokens = []
    if not tokens:
        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        #tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment).replace("\n", " ").split(' ')
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                #if we find an embedding, deal with the embedding
                if word.startswith(self.embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        print(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                    #if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                #parse word
                tokens.append([(t, weight) for t in self.tokenizer(word)["input_ids"][self.tokens_start:-1]])
    if return_tokens: return tokens

    #reshape token array to CLIP input size
    batched_tokens = []
    batch = []
    if self.start_token is not None:
        batch.append((self.start_token, 1.0, 0))
    batched_tokens.append(batch)
    for i, t_group in enumerate(tokens):
        #determine if we're going to try and keep the tokens in a single batch
        is_large = len(t_group) >= self.max_word_length

        while len(t_group) > 0:
            if len(t_group) + len(batch) > self.max_length - 1:
                remaining_length = self.max_length - len(batch) - 1
                #break word in two and add end token
                if is_large:
                    batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                    batch.append((self.end_token, 1.0, 0))
                    t_group = t_group[remaining_length:]
                #add end token and pad
                else:
                    batch.append((self.end_token, 1.0, 0))
                    if self.pad_to_max_length:
                        batch.extend([(pad_token, 1.0, 0)] * (remaining_length))
                #start new batch
                batch = []
                if self.start_token is not None:
                    batch.append((self.start_token, 1.0, 0))
                batched_tokens.append(batch)
            else:
                batch.extend([(t,w,i+1) for t,w in t_group])
                t_group = []

    #fill last batch
    batch.append((self.end_token, 1.0, 0))
    if self.pad_to_max_length:
        batch.extend([(pad_token, 1.0, 0)] * (self.max_length - len(batch)))

    if not return_word_ids:
        batched_tokens = [[(t, w) for t, w,_ in x] for x in batched_tokens]

    return batched_tokens

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

def inject_code(original_func, data, mode='a'):
    # Get the source code of the original function
    original_source = inspect.getsource(original_func)

    # Split the source code into lines
    lines = original_source.split("\n")

    for item in data:
        # Find the line number of the target line
        target_line_number = None
        for i, line in enumerate(lines):
            if item['target_line'] not in line: continue
            target_line_number = i + 1
            if item.get("mode","insert") == "replace":
                lines[i] = lines[i].replace(item['target_line'], item['code_to_insert'])
                break

            # Find the indentation of the line where the new code will be inserted
            indentation = ''
            for char in line:
                if char == ' ':
                    indentation += char
                else:
                    break
            
            # Indent the new code to match the original
            code_to_insert = item['code_to_insert']
            if item.get("dedent",True):
                code_to_insert = dedent(item['code_to_insert'])
            code_to_insert = indent(code_to_insert, indentation)

            break

        # Insert the code to be injected after the target line
        if item.get("mode","insert") == "insert" and target_line_number is not None:
            lines.insert(target_line_number, code_to_insert)

    # Recreate the modified source code
    modified_source = "\n".join(lines)
    modified_source = dedent(modified_source.strip("\n"))
    return write_to_file_and_return_fn(original_func, modified_source, mode)

def write_to_file_and_return_fn(original_func, source:str, mode='a'):
    # Write the modified source code to a temporary file so the
    # source code and stack traces can still be viewed when debugging.
    custom_name = ".patches.py"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    temp_file_path = os.path.join(current_dir, custom_name)
    with open(temp_file_path, mode) as temp_file:
        temp_file.write(source)
        temp_file.write("\n")
        temp_file.flush()

        MODULE_PATH = temp_file.name
        MODULE_NAME = __name__.split('.')[0].replace('-','_') + "_patch_modules"
        spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Retrieve the modified function from the module
        modified_function = getattr(module, original_func.__name__)

    # Adapted from https://stackoverflow.com/a/49077211
    def copy_func(f, globals=None, module=None, code=None, update_wrapper=True):
        if globals is None: globals = f.__globals__
        if code is None: code = f.__code__
        g = types.FunctionType(code, globals, name=f.__name__,
                            argdefs=f.__defaults__, closure=f.__closure__)
        if update_wrapper: g = functools.update_wrapper(g, f)
        if module is not None: g.__module__ = module
        g.__kwdefaults__ = copy(f.__kwdefaults__)
        return g
        
    return copy_func(original_func, code=modified_function.__code__, update_wrapper=False)


hook_all(hooks=[
            # hook_tokenize_with_weights(),
            hook_load_torch_file(),
])