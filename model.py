# Merge image encoder and fuse module to create an ID Encoder
# send multiple ID images, we can directly obtain the updated text encoder containing a stacked ID embedding

import torch
import torch.nn as nn
# from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
# from transformers.models.clip.configuration_clip import CLIPVisionConfig
# from transformers import PretrainedConfig
from comfy.clip_model import ACTIVATIONS
from comfy.clip_model import CLIPVisionModelProjection
from comfy.ops import manual_cast

VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}
act_fn = None
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, op: manual_cast, dtype, device, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = op.LayerNorm(in_dim, device=device, dtype=dtype)
        self.fc1 = op.Linear(in_dim, hidden_dim, device=device, dtype=dtype)
        self.fc2 = op.Linear(hidden_dim, out_dim, device=device, dtype=dtype)
        self.use_residual = use_residual
        global act_fn
        self.act_fn = act_fn 
        # self.layernorm = nn.LayerNorm(in_dim)
        # self.fc1 = nn.Linear(in_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, out_dim)
        # self.use_residual = use_residual
        # self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class FuseModule(nn.Module):
    def __init__(self, embed_dim, op: manual_cast, dtype, device):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False, op=op, device=device, dtype=dtype)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True, op=op, device=device, dtype=dtype)
        self.layer_norm = op.LayerNorm(embed_dim, device=device, dtype=dtype)
        # self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(
        self,
        prompt_embeds,
        id_embeds,
        class_tokens_mask,
    ) -> torch.Tensor:
        # id_embeds shape: [b, max_num_inputs, 1, 2048]
        id_embeds = id_embeds.to(prompt_embeds.dtype)
        num_inputs = class_tokens_mask.sum().unsqueeze(0) # TODO: check for training case
        batch_size, max_num_inputs = id_embeds.shape[:2]
        # seq_length: 77
        seq_length = prompt_embeds.shape[1]
        # flat_id_embeds shape: [b*max_num_inputs, 1, 2048]
        flat_id_embeds = id_embeds.view(
            -1, id_embeds.shape[-2], id_embeds.shape[-1]
        )
        # valid_id_mask [b*max_num_inputs]
        valid_id_mask = (
            torch.arange(max_num_inputs, device=flat_id_embeds.device)[None, :]
            < num_inputs[:, None]
        )
        valid_id_embeds = flat_id_embeds[valid_id_mask.flatten()]

        prompt_embeds = prompt_embeds.view(-1, prompt_embeds.shape[-1])
        class_tokens_mask = class_tokens_mask.view(-1)
        valid_id_embeds = valid_id_embeds.view(-1, valid_id_embeds.shape[-1])
        # slice out the image token embeddings
        image_token_embeds = prompt_embeds[class_tokens_mask]
        stacked_id_embeds = self.fuse_fn(image_token_embeds, valid_id_embeds)
        assert class_tokens_mask.sum() == stacked_id_embeds.shape[0], f"{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}"
        prompt_embeds.masked_scatter_(class_tokens_mask[:, None], stacked_id_embeds.to(prompt_embeds.dtype))
        updated_prompt_embeds = prompt_embeds.view(batch_size, seq_length, -1)
        return updated_prompt_embeds

# class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
#     def __init__(self, config_dict, dtype, device, operations):
#         super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
class PhotoMakerIDEncoder(CLIPVisionModelProjection):
    def __init__(self, config_dict, dtype, device, op: manual_cast):
        super().__init__(config_dict, dtype, device, op)
        intermediate_activation = config_dict["hidden_act"]
        global act_fn
        act_fn = ACTIVATIONS[intermediate_activation]
        self.visual_projection_2 = op.Linear(1024, 1280, bias=False, device=device, dtype=dtype)
        # self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        # self.vision_model = self.vision_model.to(device=device,dtype=dtype)
        # self.visual_projection = self.visual_projection.to(device=device,dtype=dtype)
        # self.visual_projection_2 = self.visual_projection_2.to(device=device,dtype=dtype)
        self.fuse_module = FuseModule(2048, op, device=device, dtype=dtype)

    # def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask):
    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask, *args, **kwargs):
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(b * num_inputs, c, h, w)

        # CLIPVisionModelWithProjection
        # x = self.vision_model(id_pixel_values, output_hidden_states=True)
        # shared_id_embeds = x[1].to(device=self.device,dtype=self.dtype)

        x = self.vision_model(id_pixel_values, **kwargs)
        shared_id_embeds = x[2]
        # shared_id_embeds = self.vision_model(id_pixel_values)[2]
        # shared_id_embeds = self.vision_model(id_pixel_values, intermediate_output=kwargs['intermediate_output'])[1]
        # shared_id_embeds = self.vision_model(id_pixel_values, kwargs['intermediate_output'])[1]
        # shared_id_embeds = self.vision_model(*args, **kwargs))[1]
        # shared_id_embeds = self.vision_model(id_pixel_values)[1]
        # shared_id_embeds = shared_id_embeds[2] if shared_id_embeds[2] is not None else shared_id_embeds[0]

        id_embeds = self.visual_projection(shared_id_embeds)
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)

        id_embeds = id_embeds.view(b, num_inputs, 1, -1)
        id_embeds_2 = id_embeds_2.view(b, num_inputs, 1, -1)

        id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1)
        prompt_embeds = prompt_embeds[0].to(device=id_embeds.device)
        class_tokens_mask = class_tokens_mask.to(device=id_embeds.device)
        updated_prompt_embeds = self.fuse_module(prompt_embeds, id_embeds, class_tokens_mask)

        return (x[0], x[1], updated_prompt_embeds)

        # CLIPVisionModelWithProjection
        # return (x.last_hidden_state, x.hidden_states[0], updated_prompt_embeds)


if __name__ == "__main__":
    PhotoMakerIDEncoder()